#include "model.h"
#include <stdlib.h>
#include <shrUtils.h>
#include <cutil_inline.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <iostream>

#include "kernel.cu"

using namespace std;

extern clock_t kernel_time = 0;
extern clock_t cpu_reduction_time = 0;
extern clock_t gpu_ini_time = 0;

clock_t start;

void dummyMapperScan(KColumn* columns, KTransaction* transactions, KNode* nodes, int* valid_counts, int transaction_count, int column_count, int node_count, int num_threads);

// Host data
KColumn* h_columns = NULL;
KTransaction* h_transactions = NULL;
KNode* h_nodes = NULL;
int* h_valid_counts = NULL;

// Device data
KColumn* d_columns = NULL;
KTransaction* d_transactions = NULL;
KNode* d_nodes = NULL;
int* d_valid_counts = NULL;

// Transaction and column initialized flag
bool ini_flag = false;

// Global counters
int node_count = 0;

// Generate a GPU data typed node array from the current node path
void getNode(Node* node)
{
    // count node number
    node_count = 0;
    Node* current = node;
    while (current != NULL)
    {
        node_count++;
        current = current->parent;
    }
    //cout << node_count << endl;
    h_nodes = new KNode[node_count];
    current = node;
    for (int i = 0; i < node_count; i++)
    {
        h_nodes[i].column_id = current->column != NULL ? current->column->id : -1;
        h_nodes[i].option_id = current->option != NULL ? current->option->id : -1;
        h_nodes[i].gain_ratio = current->gain_ratio;
        h_nodes[i].count = current->count;
        current = current->parent;
    }
}

// Translate transactions to GPU friendly data structure
void getTransaction(Transaction* transactions, int transaction_count, int column_count)
{
    h_transactions = new KTransaction[transaction_count];
    for (int i = 0; i < transaction_count; i++)
    {
        h_transactions[i].attribute_count = column_count;
        for (int j = 0; j < column_count; j++)
        {
            h_transactions[i].attributes[j] = transactions[i].attributes[j];
        }
        h_transactions[i].result = transactions[i].result;
    }
}

// Translate columns to GPU friendly data structure
// Spawn clones of KColumn collection
void getColumn(Column* columns, int column_count, int clone_count)
{
    h_columns = new KColumn[clone_count * column_count];
    for (int i = 0; i < clone_count; i++)
    {
        for (int j = 0; j < column_count; j++)
        {
            h_columns[i * column_count + j].id = columns[j].id;
            h_columns[i * column_count + j].option_count = columns[j].option_count;
            for (int k = 0; k < columns[j].option_count; k++)
            {
                h_columns[i * column_count + j].options[k].column_id = j;
                h_columns[i * column_count + j].options[k].id = columns[j].options[k].id;
                h_columns[i * column_count + j].options[k].no_count  = 0;
                h_columns[i * column_count + j].options[k].yes_count = 0;
            }
        }
    }
}

extern "C"
int update_gpu(Node* node, Transaction* transactions, int transaction_count, Column* columns, int column_count, int num_threads)
{
    // Compute the size of needed device memory
    int columns_size = sizeof(KColumn) * column_count * num_threads;
    int transactions_size = sizeof(KTransaction) * transaction_count;
    int valid_counts_size = sizeof(int) * num_threads;

    // Initialize fix data
    if (ini_flag == false)
    {
        // Generate GPU friendly data type
        getColumn(columns, column_count, num_threads);
        getTransaction(transactions, transaction_count, column_count);
        h_valid_counts = new int[num_threads];

        // Allocate device memory
        cutilSafeCall(cudaMalloc((void**) &d_columns, columns_size));
        cutilSafeCall(cudaMalloc((void**) &d_transactions, transactions_size));
        cutilSafeCall(cudaMalloc((void**) &d_valid_counts, valid_counts_size));

        // Copy host memory to device
        cutilSafeCall(cudaMemcpy(d_columns, h_columns, columns_size, cudaMemcpyHostToDevice));
        cutilSafeCall(cudaMemcpy(d_transactions, h_transactions, transactions_size, cudaMemcpyHostToDevice));

        ini_flag = true;
    }
    // Initialize dynamic data
    // Generate GPU friendly data type
    getNode(node);

    // Compute the size of needed device memory
    int nodes_size = sizeof(KNode) * node_count;

    // Allocate device memory
    cutilSafeCall(cudaMalloc((void**) &d_nodes, nodes_size));

    // Copy host memory to de device
    cutilSafeCall(cudaMemcpy(d_nodes, h_nodes, nodes_size, cudaMemcpyHostToDevice));
    
    // Setup execution parameters
    dim3 grid(1, 1);
    dim3 threads(num_threads, 1);

    // Excute the kernel
    mapperScan<<<grid, threads>>> (d_columns, d_transactions, d_nodes, d_valid_counts, transaction_count, column_count, node_count, num_threads);
    //dummyMapperScan(h_columns, h_transactions, h_nodes, h_valid_counts, transaction_count, column_count, node_count, num_threads);

    // Copy result from device to host
    cutilSafeCall(cudaMemcpy(h_columns, d_columns, columns_size, cudaMemcpyDeviceToHost));
    cutilSafeCall(cudaMemcpy(h_valid_counts, d_valid_counts, valid_counts_size, cudaMemcpyDeviceToHost));

    // Compute valid_count
    int valid_count = 0;
    for (int i = 0; i < num_threads; i++)
    {
        valid_count += h_valid_counts[i];
    }

    // Reduce columns' yes/no counter
    for (int i = 0; i < column_count; i++)
    {
        for (int j = 0; j < columns[i].option_count; j++)
        {
            columns[i].options[j].yes_count = 0;
            columns[i].options[j].no_count = 0;
            for (int k = 0; k < num_threads; k++)
            {
                columns[i].options[j].yes_count += h_columns[i + column_count * k].options[j].yes_count;
                columns[i].options[j].no_count  += h_columns[i + column_count * k].options[j].no_count;
            }
            //cout << columns[i].options[j].yes_count << '\t';
            //cout << columns[i].options[j].no_count  << endl;
        }
    }
    // Cleanup dynamic device data
    cutilSafeCall(cudaFree(d_nodes));
    return valid_count;
}

extern "C"
int update_gpu_2(Node* node, Transaction* transactions, int transaction_count, Column* columns, int column_count, int block_size)
{
    start = clock();
    // Compute the size of needed device memory
    int num_blocks = (int) ceil(transaction_count / (float) block_size);
    int columns_size = sizeof(KColumn) * column_count * num_blocks;
    int transactions_size = sizeof(KTransaction) * transaction_count;
    int valid_counts_size = sizeof(int) * num_blocks;

    // Initialize fix data
    if (ini_flag == false)
    {
        // Generate GPU friendly data type
        getColumn(columns, column_count, num_blocks);
        getTransaction(transactions, transaction_count, column_count);
        h_valid_counts = new int[num_blocks];

        // Allocate device memory
        cutilSafeCall(cudaMalloc((void**) &d_columns, columns_size));
        cutilSafeCall(cudaMalloc((void**) &d_transactions, transactions_size));
        cutilSafeCall(cudaMalloc((void**) &d_valid_counts, valid_counts_size));

        // Copy host memory to device
        cutilSafeCall(cudaMemcpy(d_columns, h_columns, columns_size, cudaMemcpyHostToDevice));
        cutilSafeCall(cudaMemcpy(d_transactions, h_transactions, transactions_size, cudaMemcpyHostToDevice));

        ini_flag = true;
    }
    // Initialize dynamic data
    // Generate GPU friendly data type
    getNode(node);

    // Compute the size of needed device memory
    int nodes_size = sizeof(KNode) * node_count;

    // Allocate device memory
    cutilSafeCall(cudaMalloc((void**) &d_nodes, nodes_size));

    // Copy host memory to de device
    cutilSafeCall(cudaMemcpy(d_nodes, h_nodes, nodes_size, cudaMemcpyHostToDevice));
    
    gpu_ini_time += clock() - start;
    start = clock();

    // Setup execution parameters
    dim3 grid(num_blocks, 1);
    dim3 threads(block_size, 1);

    // Excute the kernel
    mapperScan_2<<<grid, threads>>> (d_columns, d_transactions, d_nodes, d_valid_counts, transaction_count, column_count, node_count);
    //dummyMapperScan(h_columns, h_transactions, h_nodes, h_valid_counts, transaction_count, column_count, node_count, num_threads);

    // Copy result from device to host
    cutilSafeCall(cudaMemcpy(h_columns, d_columns, columns_size, cudaMemcpyDeviceToHost));
    cutilSafeCall(cudaMemcpy(h_valid_counts, d_valid_counts, valid_counts_size, cudaMemcpyDeviceToHost));

    kernel_time += clock() - start;
    start = clock();

    // Compute valid_count
    int valid_count = 0;
    for (int i = 0; i < num_blocks; i++)
    {
        valid_count += h_valid_counts[i];
    }

    // Reduce columns' yes/no counter
    for (int i = 0; i < column_count; i++)
    {
        for (int j = 0; j < columns[i].option_count; j++)
        {
            columns[i].options[j].yes_count = 0;
            columns[i].options[j].no_count = 0;
            for (int k = 0; k < num_blocks; k++)
            {
                columns[i].options[j].yes_count += h_columns[i + column_count * k].options[j].yes_count;
                columns[i].options[j].no_count  += h_columns[i + column_count * k].options[j].no_count;
            }
            //cout << columns[i].options[j].yes_count << '\t';
            //cout << columns[i].options[j].no_count  << endl;
        }
    }
    cpu_reduction_time += clock() - start;

    // Cleanup dynamic device data
    cutilSafeCall(cudaFree(d_nodes));
    return valid_count;
}

// C++ (host) update counter function
extern "C"
int update(Node* node, Transaction* transactions, int transaction_count, Column* columns, int column_count)
{
    int valid_count = 0;
	// Check each transaction if it belongs to the current node.
    for (int i = 0; i < transaction_count; i++)
    {
        // flag = true:  transaction is valid. (default)
        // flag = false: transaction is not valid.
        bool flag = true;
        Node* current = node;
		// Skip root node since it does not have an option, which means all the transactions are valid.
        while (current->option != NULL)
        {
            // Check if the current node's option id is equal to the transaction's attribute id in the given column.
            if (transactions[i].attributes[current->option->column->id] != current->option->id)
            {
                flag = false;
                break;
            }
            else
            {
                // If the transaction is valid for the current node, continue to check the parent node.
                current = current->parent;
            }
        }
        // If the transaction is valid, compute the yes and no counters for each remaining column
        if (flag)
        {
            valid_count++;
            for (int j = 0; j < column_count; j++)
            {
                if (transactions[i].result == 1)
                {
                    columns[j].options[transactions[i].attributes[j]].yes_count++;
                }
                else
                {
                    columns[j].options[transactions[i].attributes[j]].no_count++;
                }
            }
        }
    }
    //for (int i = 0; i < column_count; i++)
    //{
    //    for (int j = 0; j < columns[i].option_count; j++)
    //    {
    //        cout << columns[i].options[j].yes_count << '\t';
    //        cout << columns[i].options[j].no_count  << endl;
    //    }
    //}
    return valid_count;
}

// Initialization GPU device
extern "C"
void iniGPU()
{
    cudaSetDevice(cutGetMaxGflopsDeviceId());
}

// Exit and cleanup GPU device
extern "C"
void cleanGPU()
{
    cutilSafeCall(cudaFree(d_transactions));
    cutilSafeCall(cudaFree(d_columns));
    cutilSafeCall(cudaFree(d_valid_counts));
    cudaThreadExit();
}

/* ---------------------------------------------------------------------------- */
/* kernel wrapper test functions                                                */
/* ---------------------------------------------------------------------------- */

void integerArrayGold(int* data, int count)
{
    for (int i = 0; i < count; i++)
    {
        for (int j = 0; j < 50; j++)
        {
            for (int k = 0; k < 50; k++)
            {
                data[i] += i + j + k;
            }
        }
    }
}

extern "C"
void integer_array_performance_test(int count, int block_size)
{
    cudaSetDevice(cutGetMaxGflopsDeviceId());

    // Allocate host memory
    int* h_data = new int[count];
    int* h_result = new int[count];
    for (int i = 0; i < count; i++)
    {
        h_data[i] = 0;
    }

    // Compute size and counter
    int mem_size = sizeof(int) * count;
    //int step = (count % block_size) == 0 ? count / block_size : count / block_size + 1;

    // Device memory pointer
    int* d_data = NULL;
    // Allocate device memory
    cutilSafeCall(cudaMalloc((void**) &d_data, mem_size));
    // Copy host memory to de device
    cutilSafeCall(cudaMemcpy(d_data, h_data, mem_size, cudaMemcpyHostToDevice));

    // create and start GPU timer
    unsigned int GPUtimer = 0;
    cutilCheckError(cutCreateTimer(&GPUtimer));
    cutilCheckError(cutStartTimer(GPUtimer));

    // Setup execution parameters
    dim3 grid((int) ceil(count / (float) block_size), 1);
    dim3 threads(block_size, 1);
    // Excute the kernel
    integerArray<<<grid, threads>>> (d_data, count);

    // Copy result from device to host
    cutilSafeCall(cudaMemcpy(h_result, d_data, mem_size, cudaMemcpyDeviceToHost));

    // stop GPU timer
    cutilCheckError(cutStopTimer(GPUtimer));
    float gpu_time = cutGetTimerValue(GPUtimer);
    printf("GPU processing time: %f (ms) \n", gpu_time);
    cutilCheckError(cutDeleteTimer(GPUtimer));

    // Start CPU timer
    unsigned int CPUtimer = 0;
    cutilCheckError(cutCreateTimer(&CPUtimer));
    cutilCheckError(cutStartTimer(CPUtimer));

    // execute the CPU function
    integerArrayGold(h_data, count);

    // stop and destory CPU timer
    cutilCheckError(cutStopTimer(CPUtimer));
    float cpu_time = cutGetTimerValue(CPUtimer);
    printf("CPU processing time: %f (ms) \n", cpu_time);
    cutilCheckError(cutDeleteTimer(CPUtimer));

    // Print out results
    for (int i = count - 10; i < count; i++)
    {
        cout << h_result[i] << '\t';
    }
    cout << endl;
    cout << endl;
    cout << endl;

    // Cleanup
    cutilSafeCall(cudaFree(d_data));
    cudaThreadExit();
}

void floatArrayGold(float* data, int count)
{
    for (int i = 0; i < count; i++)
    {
        for (int j = 0; j < 50; j++)
        {
            for (int k = 0; k < 50; k++)
            {
                data[i] += i * 0.1 + j * 0.1 + k * 0.1;
            }
        }
    }
}

extern "C"
void float_array_performance_test(int count, int block_size)
{
    cudaSetDevice(cutGetMaxGflopsDeviceId());

    // Allocate host memory
    float* h_data = new float[count];
    float* h_result = new float[count];
    for (int i = 0; i < count; i++)
    {
        h_data[i] = 1.1;
    }

    // Compute size and counter
    int mem_size = sizeof(float) * count;
    //int step = (count % block_size) == 0 ? count / block_size : count / block_size + 1;

    // Device memory pointer
    float* d_data = NULL;
    // Allocate device memory
    cutilSafeCall(cudaMalloc((void**) &d_data, mem_size));
    // Copy host memory to de device
    cutilSafeCall(cudaMemcpy(d_data, h_data, mem_size, cudaMemcpyHostToDevice));

    // create and start GPU timer
    unsigned int GPUtimer = 0;
    cutilCheckError(cutCreateTimer(&GPUtimer));
    cutilCheckError(cutStartTimer(GPUtimer));

    // Setup execution parameters
    dim3 grid((int) ceil(count / (float) block_size), 1);
    dim3 threads(block_size, 1);
    // Excute the kernel
    floatArray<<<grid, threads>>> (d_data, count);

    // Copy result from device to host
    cutilSafeCall(cudaMemcpy(h_result, d_data, mem_size, cudaMemcpyDeviceToHost));

    // stop GPU timer
    cutilCheckError(cutStopTimer(GPUtimer));
    float gpu_time = cutGetTimerValue(GPUtimer);
    printf("GPU processing time: %f (ms) \n", gpu_time);
    cutilCheckError(cutDeleteTimer(GPUtimer));

    // Start CPU timer
    unsigned int CPUtimer = 0;
    cutilCheckError(cutCreateTimer(&CPUtimer));
    cutilCheckError(cutStartTimer(CPUtimer));

    // execute the CPU function
    floatArrayGold(h_data, count);

    // stop CPU timer
    cutilCheckError(cutStopTimer(CPUtimer));
    float cpu_time = cutGetTimerValue(CPUtimer);
    printf("CPU processing time: %f (ms) \n", cpu_time);
    cutilCheckError(cutDeleteTimer(CPUtimer));

    // Print out results
    for (int i = count - 10; i < count; i++)
    {
        cout << h_data[i] << '\t';
    }
    cout << endl;
    cout << endl;
    cout << endl;

    // Cleanup
    cutilSafeCall(cudaFree(d_data));
    cudaThreadExit();
}

/* ----------------------------------------- */
/* Dummy kernel                              */
/* ----------------------------------------- */

void dummyMapperScan(KColumn* columns, KTransaction* transactions, KNode* nodes, int* valid_counts, int transaction_count, int column_count, int node_count, int num_threads)
{
    int idx = 0;

    // Cleanup yes, no and valid counters
    for (int i = idx * column_count; i < num_threads * column_count && i < (idx + 1) * column_count; i++)
    {
        for (int j = 0; j < columns[i].option_count; j++)
        {
            columns[i].options[j].yes_count = 0;
            columns[i].options[j].no_count = 0;
        }
    }
    if (idx < num_threads)
    {
        valid_counts[idx] = 0;
    }

    // Iterate through each transaction
    int step = transaction_count % num_threads == 0 ? transaction_count / num_threads : transaction_count / num_threads + 1;
    for (int i = idx * step; i < transaction_count && i < (idx + 1) * step; i++)
    {
        bool flag = true;
        for (int j = 0; j < node_count && nodes[j].option_id >= 0; j++)
        {
            if (transactions[i].attributes[nodes[j + 1].column_id] != nodes[j].option_id)
            {
                flag = false;
                break;
            }
        }
        if (flag)
        {
            valid_counts[idx]++;
            for (int j = idx * column_count; j < num_threads * column_count && j < (idx + 1) * column_count; j++)
            {
                if (transactions[i].result == 1)
                {
                    columns[j].options[transactions[i].attributes[j % column_count]].yes_count++;
                }
                else
                {
                    columns[j].options[transactions[i].attributes[j % column_count]].no_count++;
                }
            }
        }
    }
}