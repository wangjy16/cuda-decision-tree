#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <time.h>
#include <kernel.h>
#include "model.h"

#include <cutil_inline.h>

extern "C" void iniGPU();
extern "C" void cleanGPU();
extern "C" void array_add_test(int input);
extern "C" int update(Node* node, Transaction* transactions, int transaction_count, Column* columns, int column_count);
extern "C" int update_gpu(Node* node, Transaction* transactions, int transaction_count, Column* columns, int column_count, int num_threads);
extern "C" int update_gpu_2(Node* node, Transaction* transactions, int transaction_count, Column* columns, int column_count, int block_size);
extern "C" void integer_array_performance_test(int count, int num_threads);
extern "C" void float_array_performance_test(int count, int num_threads);

extern clock_t kernel_time;
extern clock_t cpu_reduction_time;
extern clock_t gpu_ini_time;

using namespace std;

Tree t;

void info_single_test()
{
    cout << t.info(9, 0) << endl;
}

void gain_test()
{
    double age[3][2] = {{2, 3},
                        {4, 0},
                        {3, 2}};
    cout << "age gain: " << t.gain(age, 3) << endl;
}

void split_info_test()
{
    double age[3][2] = {{2, 3},
                        {4, 0},
                        {3, 2}};
    cout << "age gain: " << t.splitInfo(age, 14, 3) << endl;
}

void gain_ratio_test()
{
    double age[3][2] = {{2, 3},
                        {4, 0},
                        {3, 2}};
    cout << "age gain: " << t.gainRatio(age, 3) << endl;
}

void column_gain_ratio_test()
{
    Column* columns = t.generateColumn(7, 2, 5);
    cout << t.gainRatio(columns) << endl;
}

void generate_column_test()
{
    Column *columns = t.generateColumn(30, 2, 5);
    for (int i = 0; i < 30; i++)
    {
        for (int j = 0; j < columns[i].option_count; j++)
        {
            cout << "column: " << columns[i].id << "\t" << "option: " << columns[i].options[j].id << endl;
        }
    }
}

void generate_transaction_table()
{
    int column_count = 2;
    int transaction_count = 10;
    Column *columns = t.generateColumn(column_count, 2, 5);
    cout << "Transaction #\t";
    for (int i = 0; i < column_count; i++)
    {
        cout << "column " << columns[i].id << " (" << columns[i].option_count << ")\t";
    }
    cout << "Result";
    Transaction *transactions = t.generateTransaction(columns, column_count, transaction_count);
    int id = 0;
    for (int i = 0; i < transaction_count; i++)
    {
        cout << endl;
        cout << id++ << "\t\t";
        for (int j = 0; j < column_count; j++)
        {
            cout << "option " << transactions[i].attributes[j] << "\t";
        }
        cout << transactions[i].result;
    }
    cout << endl;
}

void build_single_node()
{
    int column_count = 3;
    int transaction_count = 1000 * 10;
    Column* columns = t.generateColumn(column_count, 2, 5);
    Transaction* transactions = t.generateTransaction(columns, column_count, transaction_count);
    Node* node = new Node;
    t.iniNode(node);
    t.scan(node, transactions, transaction_count, columns, column_count, 0);
    cout << node->column->id << endl;
    cout << node->gain_ratio << endl;
}

void build_full_tree_and_dfs_travel()
{
    clock_t start, gpu_time;
    int column_count = 3;
    int transaction_count = 4096 * 256;
    Column* columns = t.generateColumn(column_count, 2, 2);
    Transaction* transactions = t.generateTransaction(columns, column_count, transaction_count);
    Node* node = new Node;
    t.iniNode(node);
    // GPU building
    start = clock();
    t.build(node, transactions, transaction_count, columns, column_count, 1);
    gpu_time = clock() - start;
    cout << "Total time: " << (float) gpu_time << " (ms)" << endl;
    cout << "Kernel time: " << (float) kernel_time << " (ms)" << endl;
    cout << "gpu ini time: " << (float) gpu_ini_time << " (ms)" << endl;
    cout << "cpu reduction time: " << (float) cpu_reduction_time << " (ms)" << endl;
    cout << "Parent  \t" << "Option  \t" << "Current \t" << "Gain Info\t" << "Trans Count" << endl;
    t.DFSTravel(node);
}
void build_full_tree_from_file_and_dfs_travel()
{
    char* path = "C:/Users/Tong/Desktop/Data/1000_3_2_3.txt";
    t.readTable(path);
    Node* node = new Node;
    t.iniNode(node);
    t.build(node, t.transactions, t.transactionCount, t.columns, t.columnCount, false);
    cout << "Parent  \t" << "Option  \t" << "Current \t" << "Gain Info\t" << "Trans Count" << endl;
    t.DFSTravel(node);
}

void file_io_test()
{
    t.readTable("C:/Users/Tong/Desktop/Data/test.txt");
}

void string_split_wrapper_test()
{
    char str[] = "\tasdf\tqwer\t";
    vector<char*> vec = t.split(str, "\t");
    for (unsigned int i = 0; i < vec.size(); i++)
    {
        cout << vec[i] << endl;
    }
}

void c_strtok_test()
{
    char str[] = "asdf\tqwer\t";
    char* token = strtok((char*) "asdf", "\t");
    cout << token << endl;
}

void string_to_integer_test()
{
    double a = atof("12.32");
    cout << a + 1 << endl;
}

void column_id_parser_test()
{
    char token[] = "column 2(2)";
    int columnId = atoi(t.split(t.split(token, " ")[1], "(")[0]);
    cout << columnId << endl;
}

void option_count_parser_test()
{
    char token[] = "column 2(2)";
    int optionCount = atoi(t.split(t.split(t.split(token, " ")[1], "(")[1], ")")[0]);
    cout << optionCount << endl;
}

void read_table_from_file()
{
    char* path = "C:/Users/Tong/Desktop/Data/test.txt";
    t.readTable(path);
    for (int i = 0; i < t.columnCount; i++)
    {
        cout << "column " << t.columns[i].id << " (" << t.columns[i].option_count << ")\t";
    }
    cout << "Result";
    for (int i = 0; i < 10; i++)
    {
        cout << endl;
        cout << i << "\t";
        for (int j = 0; j < t.transactions[i].attribute_count; j++)
        {
            cout << "option " << t.transactions[i].attributes[j] << "\t";
        }
        cout << t.transactions[i].result;
    }
    cout << endl << "column count: " << t.columnCount << endl;
    cout << endl << "transactions count: " << t.transactionCount << endl;
}

KTransaction* get_kernel_transaction(Transaction* transactions, int transaction_count, Column* columns, int column_count)
{
    // code
    KTransaction* ktransactions = new KTransaction[transaction_count];
    for (int i = 0; i < transaction_count; i++)
    {
        ktransactions[i].attribute_count = column_count;
        for (int j = 0; j < column_count; j++)
        {
            ktransactions[i].attributes[j] = transactions[i].attributes[j];
        }
        ktransactions[i].result = transactions[i].result;
    }
    // end code
    return ktransactions;
}
void get_kernel_transaction_test()
{
    int column_count = 2;
    int transaction_count = 50;
    Column *columns = t.generateColumn(column_count, 2, 5);
    cout << "Transaction #\t";
    for (int i = 0; i < column_count; i++)
    {
        cout << "column " << columns[i].id << " (" << columns[i].option_count << ")\t";
    }
    cout << "Result";
    Transaction* o_transactions = t.generateTransaction(columns, column_count, transaction_count);
    KTransaction* transactions = get_kernel_transaction(o_transactions, transaction_count, columns, column_count);
    int id = 0;
    for (int i = 0; i < transaction_count; i++)
    {
        cout << endl;
        cout << id++ << "\t\t";
        for (int j = 0; j < column_count; j++)
        {
            cout << "option " << transactions[i].attributes[j] << "\t";
        }
        cout << transactions[i].result;
    }
    cout << endl;
}

KColumn* get_kernel_column(Column* columns, int column_count)
{
    // code
    KColumn* kcolumns = new KColumn[column_count];
    for (int i = 0; i < column_count; i++)
    {
        kcolumns[i].id = columns[i].id;
        kcolumns[i].option_count = columns[i].option_count;
        for (int j = 0; j < columns[i].option_count; j++)
        {
            kcolumns[i].options[j].column_id = i;
            kcolumns[i].options[j].id = columns[i].options[j].id;
            kcolumns[i].options[j].no_count = 0;
            kcolumns[i].options[j].yes_count = 0;
        }
    }
    // end code
    return kcolumns;
}
void get_kernel_column_test()
{
    int column_count = 10;
    Column* o_columns = t.generateColumn(column_count, 2, 5);
    KColumn* columns = get_kernel_column(o_columns, column_count);
    for (int i = 0; i < column_count; i++)
    {
        for (int j = 0; j < columns[i].option_count; j++)
        {
            cout << "column: " << columns[i].id << "\t" << "option: " << columns[i].options[j].id << endl;
        }
    }
}

void get_kernel_node_array()
{
    // Initialization
    int column_count = 5;
    int transaction_count = 10000;
    Column* columns = t.generateColumn(column_count, 2, 5);
    Transaction* transactions = t.generateTransaction(columns, column_count, transaction_count);
    Node* node = new Node;
    t.iniNode(node);
    t.build(node, transactions, transaction_count, columns, column_count, true);
    
    // Buildd node array
    int node_count = 0;
    Node* current = node->children->children->children->children;
    while (current != NULL)
    {
        node_count++;
        //cout << current->column->id << endl;
        current = current->parent;
    }
    KNode* nodes = new KNode[node_count];
    current = node->children->children->children->children;
    //cout << node_count << endl;
    for (int i = 0; i < node_count; i++)
    {
        nodes[i].column_id = current->column != NULL ? current->column->id : -1;
        nodes[i].count = current->count;
        nodes[i].gain_ratio = current->gain_ratio;
        nodes[i].option_id = current->option != NULL ? current->option->id : -1;
        current = current->parent;
    }
    for (int i = 0; i < node_count; i++)
    {
        cout << nodes[i].column_id << endl;
    }
}
//#include <shrUtils.h>
//#include "kernel.cu"
// GPU functions
void integer_array_different_block_size_test()
{
    integer_array_performance_test(10000, 2);
    integer_array_performance_test(10000, 4);
    integer_array_performance_test(10000, 8);
    integer_array_performance_test(10000, 16);
    integer_array_performance_test(10000, 32);
    integer_array_performance_test(10000, 64);
    integer_array_performance_test(10000, 128);
    integer_array_performance_test(10000, 256);
}
void float_array_different_block_size_test()
{
    float_array_performance_test(10000, 4);
    float_array_performance_test(10000, 8);
    float_array_performance_test(10000, 16);
    float_array_performance_test(10000, 32);
    float_array_performance_test(10000, 64);
    float_array_performance_test(10000, 128);
    float_array_performance_test(10000, 256);
}
void kernel_type_size_test()
{
    cout << "KTransaction:\t" << sizeof(KTransaction) << endl;
    cout << "KColumn:\t" << sizeof(KColumn) << endl;
    cout << "KOption:\t" << sizeof(KOption) << endl;
    cout << "KNode:\t" << sizeof(KNode) << endl;
}
void update_test()
{
    int column_count = 10;
    int transaction_count = 41235;
    int num_threads = 16;
    int block_size = 32;
    Column* columns = t.generateColumn(column_count, 2, 5);
    Transaction* transactions = t.generateTransaction(columns, column_count, transaction_count);
    Node* node = new Node;
    t.iniNode(node);
    cout << update(node, transactions, transaction_count, columns, column_count);
    //cout << update_gpu(node, transactions, transaction_count, columns, column_count, num_threads);
    //cout << update_gpu_2(node, transactions, transaction_count, columns, column_count, block_size);
    cout << endl;
}
void gpu_cpu_build_compare()
{
    clock_t start, gpu_time, cpu_time;
    Node* gpu_node, * cpu_node;
    int column_count = 3;
    int transaction_count = 1000 * 500;
    Column* columns = t.generateColumn(column_count, 2, 2);
    Transaction* transactions = t.generateTransaction(columns, column_count, transaction_count);
    // GPU building
    gpu_node = new Node;
    t.iniNode(gpu_node);
    start = clock();
    t.build(gpu_node, transactions, transaction_count, columns, column_count, 2);
    gpu_time = clock() - start;
    cout << "GPU building time: " << (float) gpu_time << " (ms)" << endl;
    // CPU building
    cpu_node = new Node;
    t.iniNode(cpu_node);
    start = clock();
    t.build(cpu_node, transactions, transaction_count, columns, column_count, 0);
    cpu_time = clock() - start;
    cout << "CPU building time: " << (float) cpu_time << " (ms)" << endl;
    // Compare
    cout << "Speedup: " << (float) cpu_time / (float) gpu_time << endl;
    cout << "Parent  \t" << "Option  \t" << "Current \t" << "Gain Info\t" << "Trans Count" << endl;
    t.DFSTravel(gpu_node);
}

int main(int argc, char** argv)
{
    //info_single_test();
    //gain_test();
    //split_info_test();
    //gain_ratio_test();
    //generate_column_test();
    //generate_transaction_table();
    //build_single_node();
    //column_gain_ratio_test();
    //build_full_tree_and_dfs_travel();
    //file_io_test();
    //string_split_wrapper_test();
    //c_strtok_test();
    //string_to_integer_test();
    //column_id_parser_test();
    //read_table_from_file();
    //option_count_parser_test();
    //build_full_tree_from_file_and_dfs_travel();
    //get_kernel_transaction_test();
    //get_kernel_column_test();
    //get_kernel_node_array();
    //kernel_type_size_test();
    //update_test();
    //integer_array_different_block_size_test();
    //float_array_different_block_size_test();
    gpu_cpu_build_compare();
}