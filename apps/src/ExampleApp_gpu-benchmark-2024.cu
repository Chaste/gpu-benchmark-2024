/*

Copyright (c) 2005-2023, University of Oxford.
All rights reserved.

University of Oxford means the Chancellor, Masters and Scholars of the
University of Oxford, having an administrative office at Wellington
Square, Oxford OX1 2JD, UK.

This file is part of Chaste.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
 * Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
 * Neither the name of the University of Oxford nor the names of its
   contributors may be used to endorse or promote products derived from this
   software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

/**
 * @file
 *
 * This file gives an example of how you can create your own executable
 * in a user project.
 */

#include <iostream>
#include <string>

#include "ExecutableSupport.hpp"
#include "Exception.hpp"
#include "PetscTools.hpp"
#include "PetscException.hpp"

#include "flamegpu/flamegpu.h"

#include "GPUModifier.cuh"
#include "NodesOnlyMesh.hpp"
#include "UniformCellCycleModel.hpp"
#include "OffLatticeSimulation.hpp"
#include "GeneralisedLinearSpringForce.hpp"
#include "CellsGenerator.hpp"
#include "TransitCellProliferativeType.hpp"
#include "SmartPointers.hpp"
#include "SimulationTime.hpp"

#include "Hello_gpu-benchmark-2024.hpp"

FLAMEGPU_AGENT_FUNCTION(test_do_nothing, flamegpu::MessageNone, flamegpu::MessageNone) {
    return flamegpu::ALIVE;
}

FLAMEGPU_INIT_FUNCTION(test_simple_force_create_agents) {
  // Retrieve the host agent tools for agent sheep in the default state
  flamegpu::HostAgentAPI cell = FLAMEGPU->agent("cell");

  // Create 10 new cell agents
  for (int i = 0; i < 3; ++i) {
      flamegpu::HostNewAgentAPI new_cell = cell.newAgent();
      new_cell.setVariable<float>("x", i * 0.5f);
      new_cell.setVariable<float>("y", i * 0.5f);
      new_cell.setVariable<float>("x_force", 0.0f);
      new_cell.setVariable<float>("y_force", 0.0f);
      new_cell.setVariable<float>("radius", 0.5f);
  }
}

FLAMEGPU_AGENT_FUNCTION(test_output_location, flamegpu::MessageNone, flamegpu::MessageBruteForce) {
    FLAMEGPU->message_out.setVariable<float>("x", FLAMEGPU->getVariable<float>("x"));
    FLAMEGPU->message_out.setVariable<float>("y", FLAMEGPU->getVariable<float>("y"));
    FLAMEGPU->message_out.setVariable<float>("radius", FLAMEGPU->getVariable<float>("radius"));
    return flamegpu::ALIVE;
}

// Models repulsion force without division/apoptosis
FLAMEGPU_AGENT_FUNCTION(test_compute_force_meineke_spring, flamegpu::MessageBruteForce, flamegpu::MessageNone) {
    const float x = FLAMEGPU->getVariable<float>("x");
    const float y = FLAMEGPU->getVariable<float>("y");
    float x_force = 0.0;
    float y_force = 0.0;
    float radius = FLAMEGPU->getVariable<float>("radius");

    for (const auto& message : FLAMEGPU->message_in) {
        float other_x = message.getVariable<float>("x");
        float other_y = message.getVariable<float>("y");
        float other_radius = message.getVariable<float>("radius");
        
        // Compute unit distance
        float x_dist = other_x - x;
        float y_dist = other_y - y;
        float distance_between_nodes = sqrt(x_dist * x_dist + y_dist * y_dist);

        float unit_x = x_dist / distance_between_nodes;
        float unit_y = y_dist / distance_between_nodes;
        
        // Only compute force if within cutoff distance and for positive distance
        const float cutoff_length = 1.5f;
        if (distance_between_nodes < cutoff_length && distance_between_nodes > 0.0f) {

            // Compute rest length
            const float rest_length = radius + other_radius; 
            const float rest_length_final = rest_length;
            
            // TODO: Should check here if newly divided or apoptosis happening


            // Compute the force
            float overlap = distance_between_nodes - rest_length;
            bool is_closer_than_rest_length = (overlap <= 0);
            const float spring_stiffness = 15.0f;
            const float multiplication_factor = 1.0f;

            
            // A reasonably stable simple force law
            if (is_closer_than_rest_length) //overlap is negative
            {
                //assert(overlap > -rest_length_final);
                x_force += multiplication_factor * spring_stiffness * unit_x * rest_length_final* log(1.0 + overlap/rest_length_final);
                y_force  = multiplication_factor * spring_stiffness * unit_y * rest_length_final* log(1.0 + overlap/rest_length_final);
            }
            else
            {
                double alpha = 5.0;
                x_force += multiplication_factor * spring_stiffness * unit_x * overlap * exp(-alpha * overlap/rest_length_final);
                y_force += multiplication_factor * spring_stiffness * unit_y * overlap * exp(-alpha * overlap/rest_length_final);
            }
        }

        
    }

    FLAMEGPU->setVariable<float>("x_force", x_force);        
    FLAMEGPU->setVariable<float>("y_force", y_force);        

    return flamegpu::ALIVE;
}

typedef struct ResultsRow {
    std::string type;
    double box_size;
    double run_time;
} ResultsRow;

void WriteResultsToFile(std::vector<ResultsRow> results, std::string fileName) {
    std::ofstream results_file(fileName);
    for (auto& row : results) {
        results_file << row.type << ", " << row.box_size << ", " << row.run_time << "\n";
    }
    std::cout << "Results written to " << fileName << "\n";
}

void PerformGPUSim(const double size_of_box, std::vector<ResultsRow>& results) {
    
    std::cout << "Starting GPU sim with box size: " << size_of_box << "\n";
    auto start_time = std::chrono::high_resolution_clock::now();
    
    SimulationTime::Instance()->SetStartTime(0.0);
    unsigned cells_across = size_of_box * 1.52;
    double scaling = size_of_box/(double(cells_across-1));

    // Create a simple 3D NodeBasedCellPopulation consisting of cells evenly spaced in a regular grid
    std::vector<Node<2>*> nodes;
    unsigned index = 0;
    for (unsigned i=0; i<cells_across; i++)
    {
        for (unsigned j=0; j<cells_across; j++)
        {
            nodes.push_back(new Node<2>(index, false,  (double) i * scaling , (double) j * scaling));
            index++;
        }
    }

    NodesOnlyMesh<2> mesh;
    mesh.ConstructNodesWithoutMesh(nodes, 1.5);

    std::vector<CellPtr> cells;
    MAKE_PTR(TransitCellProliferativeType, p_transit_type);
    CellsGenerator<UniformCellCycleModel, 2> cells_generator;
    cells_generator.GenerateBasicRandom(cells, mesh.GetNumNodes(), p_transit_type);

    NodeBasedCellPopulation<2> node_based_cell_population(mesh, cells);
    //node_based_cell_population.AddCellPopulationCountWriter<CellProliferativeTypesCountWriter>();

    // Set up cell-based simulation
    OffLatticeSimulation<2> simulator(node_based_cell_population);
    simulator.SetOutputDirectory("GPUNodeBased");
    simulator.SetSamplingTimestepMultiple(12);
    simulator.SetEndTime(1.0);

    MAKE_PTR(GPUModifier<2>, gpuModifier);
    simulator.AddSimulationModifier(gpuModifier);

    // Run simulation
    simulator.Solve();

    // Avoid memory leak
    for (unsigned i=0; i<nodes.size(); i++)
    {
        delete nodes[i];
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    ResultsRow row;
    row.type = "gpu";
    row.box_size = size_of_box;
    row.run_time = duration.count();
    results.push_back(row);
}

void PerformCPUSim(const double size_of_box, std::vector<ResultsRow>& results) {
    
    std::cout << "Starting CPU sim with box size: " << size_of_box << "\n";
    auto start_time = std::chrono::high_resolution_clock::now();
    SimulationTime::Instance()->SetStartTime(0.0);
    unsigned cells_across = size_of_box * 1.52;
    double scaling = size_of_box/(double(cells_across-1));

    // Create a simple 3D NodeBasedCellPopulation consisting of cells evenly spaced in a regular grid
    std::vector<Node<2>*> nodes;
    unsigned index = 0;
    for (unsigned i=0; i<cells_across; i++)
    {
        for (unsigned j=0; j<cells_across; j++)
        {
            nodes.push_back(new Node<2>(index, false,  (double) i * scaling , (double) j * scaling));
            index++;
        }
    }

    NodesOnlyMesh<2> mesh;
    mesh.ConstructNodesWithoutMesh(nodes, 1.5);

    std::vector<CellPtr> cells;
    MAKE_PTR(TransitCellProliferativeType, p_transit_type);
    CellsGenerator<UniformCellCycleModel, 2> cells_generator;
    cells_generator.GenerateBasicRandom(cells, mesh.GetNumNodes(), p_transit_type);

    NodeBasedCellPopulation<2> node_based_cell_population(mesh, cells);
    //node_based_cell_population.AddCellPopulationCountWriter<CellProliferativeTypesCountWriter>();

    // Set up cell-based simulation
    OffLatticeSimulation<2> simulator(node_based_cell_population);
    simulator.SetOutputDirectory("GPUNodeBased");
    simulator.SetSamplingTimestepMultiple(12);
    simulator.SetEndTime(1.0);

    MAKE_PTR(GeneralisedLinearSpringForce<2>, springForce);
    simulator.AddForce(springForce);

    // Run simulation
    simulator.Solve();

    // Avoid memory leak
    for (unsigned i=0; i<nodes.size(); i++)
    {
        delete nodes[i];
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    ResultsRow row;
    row.type = "cpu";
    row.box_size = size_of_box;
    row.run_time = duration.count();
    results.push_back(row);
}

int main(int argc, char *argv[])
{
    // This sets up PETSc and prints out copyright information, etc.
    ExecutableSupport::StandardStartup(&argc, &argv);
    ExecutableSupport::StandardStartup(&argc, &argv);
    std::vector<double> box_sizes = {10.0, 20.0, 30.0, 40.0, 50.0, 100.0, 200.0, 300.0, 400.0, 500.0};
    std::vector<ResultsRow> results;
    for (auto box_size : box_sizes) {
        PerformGPUSim(box_size, results);
        PerformCPUSim(box_size, results);
    }
    WriteResultsToFile(results, "results.txt");
    std::cout << "Benchmark complete\n";
}
