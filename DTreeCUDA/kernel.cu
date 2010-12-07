/**
* @author Tong Pan
* #date 11/29/2010
*/

#ifndef _KERNEL_CU_
#define _KERNEL_CU_

#include "kernel.h"

__global__
void arrayAddTest(int* idata, int* odata, int count, int step, int input)
{
    int tx = threadIdx.x;
    for (int i = tx * step; i < count && i < (tx + 1) * step; i++)
    {
        idata[i] = idata[i] + input;
    }
}

__global__
void integerArray(int* data, int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count)
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

__global__
void floatArray(float* data, int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count)
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

// Cleanup yes, no and valid counters
__device__
void clean(KColumn* columns, int* valid_counts, int column_count, int num_threads, int tx)
{
    for (int i = tx * column_count; i < num_threads * column_count && i < (tx + 1) * column_count; i++)
    {
        for (int j = 0; j < columns[i].option_count; j++)
        {
            columns[i].options[j].yes_count = 0;
            columns[i].options[j].no_count = 0;
        }
    }
    if (tx < num_threads)
    {
        valid_counts[tx] = 0;
    }
}

__global__
void mapperScan(KColumn* columns, KTransaction* transactions, KNode* nodes, int* valid_counts, int transaction_count, int column_count, int node_count, int num_threads)
{
    int tx = threadIdx.x;

    clean(columns, valid_counts, column_count, num_threads, tx);

    // Iterate through each transaction
    int step = transaction_count % num_threads == 0 ? transaction_count / num_threads : transaction_count / num_threads + 1;
    for (int i = tx * step; i < transaction_count && i < (tx + 1) * step; i++)
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
            valid_counts[tx]++;
            for (int j = tx * column_count; j < num_threads * column_count && j < (tx + 1) * column_count; j++)
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

__global__
void mapperScan_2(KColumn* columns, KTransaction* transactions, KNode* nodes, int* valid_counts, int transaction_count, int column_count, int node_count)
{
    __shared__ int s_valid_counts[MAX_BLOCK_SIZE];
    __shared__ SColumn  s_columns[MAX_BLOCK_SIZE * MAX_COLUMN_COUNT];

    // Thread and block indexes
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    unsigned int gid = bid * blockDim.x + tid;

    // Fixed variables
    int s_column_start = tid * column_count;
    int s_column_end   = s_column_start + column_count;

    // Cleanup shared yes/no and valid counter
    for (int i = s_column_start; i < s_column_end; i++)
    {
        for (int j = 0; j < MAX_OPTION_COUNT; j++)
        {
            s_columns[i].options[j].yes_count = 0;
            s_columns[i].options[j].no_count  = 0;
        }
    }
    s_valid_counts[tid] = 0;
    __syncthreads();

    // Check validation of the transaction
    if (gid < transaction_count)
    {
        bool flag = true;
        for (int i = 0; i < node_count && nodes[i].option_id >= 0; i++)
        {
            if (transactions[gid].attributes[nodes[i + 1].column_id] != nodes[i].option_id)
            {
                flag = false;
                break;
            }
        }
        if (flag)
        {
            s_valid_counts[tid] = 1;
            for (int i = s_column_start; i < s_column_end; i++)
            {
                int offset = i - tid * column_count;
                if (transactions[gid].result == 1)
                {
                    s_columns[i].options[transactions[gid].attributes[offset]].yes_count = 1;
                }
                else
                {
                    s_columns[i].options[transactions[gid].attributes[offset]].no_count = 1;
                }
            }
        }
    }
    __syncthreads();

    // Do reduce in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            s_valid_counts[tid] += s_valid_counts[tid + s];
            for (int i = s_column_start; i < s_column_end; i++)
            {
                for (int j = 0; j < MAX_OPTION_COUNT; j++)
                {
                    s_columns[i].options[j].yes_count += s_columns[i + s * column_count].options[j].yes_count;
                    s_columns[i].options[j].no_count  += s_columns[i + s * column_count].options[j].no_count;;
                }
            }
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0)
    {
        valid_counts[bid] = s_valid_counts[0];
        for (int i = bid * column_count; i < (bid + 1) * column_count; i++)
        {
            int offset = i - bid * column_count;
            for (int j = 0; j < MAX_OPTION_COUNT; j++)
            {
                columns[i].options[j].yes_count = s_columns[offset].options[j].yes_count;
                columns[i].options[j].no_count  = s_columns[offset].options[j].no_count;;
            }
        }
    }
    //__syncthreads();
    //// Iterate through each transaction
    ////int step = transaction_count % block_size == 0 ? transaction_count / block_size : transaction_count / block_size + 1;
    ////for (int i = tx * step; i < transaction_count && i < (tx + 1) * step; i++)
}

#endif