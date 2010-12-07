/**
 * @author Tong Pan
 * #date 11/28/2010
 */
#ifndef _MODEL_H_
#define _MODEL_H_

#include "kernel.h"
#include <time.h>
#include <vector>

struct Transaction
{
    int* attributes;
    int  attribute_count;
    int  result;
};

struct Option;
struct Column;
struct Node;

struct Node
{
    Column* column;
    double  gain_ratio;
    Option* option;
    Node*   children;
    int     children_count;
    Node*   parent;
    int     count;
};

struct Column
{
    int     id;
    Option* options;
    int     option_count;
};

struct Option
{
    int     id;
    Column* column;
    int     yes_count;
    int     no_count;
};

class Tree
{
public:
    Tree(void);
    ~Tree(void);
    double total(double attr[][2], int option_count);
    double info(double yes, double no);
    double info(double attr[][2], double total, int option_count);
    double gain(double attr[][2], int option_count);
    double splitInfo(double attr[][2], double total, int option_count);
    double gainRatio(double attr[][2], int option_count);
    Column* generateColumn(int column_count, int option_min, int option_max);
    Transaction* generateTransaction(Column* columns, int column_count, int transaction_count);
    void scan(Node* node, Transaction* transactions, int transaction_count, Column* columns, int column_count, int gpu);
    double gainRatio(Column* column);
    void iniNode(Node* node);
    void build(Node* root, Transaction* transactions, int transaction_count , Column* columns, int column_count, int gpu);
    void DFSTravel(Node* node);
    void visit(Node* node);
    void readTable(char path[]);
    Column* columns;
    Transaction* transactions;
    std::vector<char*> split(char* str, char* delim);
    int columnCount;
    int transactionCount;
};

#endif
