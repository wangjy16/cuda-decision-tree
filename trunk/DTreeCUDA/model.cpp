#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <limits.h>

#include "model.h"

#include <cutil_inline.h>

extern "C" void iniGPU();
extern "C" void cleanGPU();
extern "C" int update(Node* node, Transaction* transactions, int transaction_count, Column* columns, int column_count);
extern "C" int update_gpu(Node* node, Transaction* transactions, int transaction_count, Column* columns, int column_count, int num_threads);
extern "C" int update_gpu_2(Node* node, Transaction* transactions, int transaction_count, Column* columns, int column_count, int block_size);

Tree::Tree(void)
    : transactions(NULL)
    , columns(NULL)
    , columnCount(0)
    , transactionCount(0)
{
    srand(1);
    iniGPU();
}

Tree::~Tree(void)
{
    cleanGPU();
}

double Tree::total(double attr[][2], int option_count)
{
    double total = 0;
    for (int i = 0; i < option_count; i++)
    {
        total += attr[i][0];
        total += attr[i][1];
    }
    return total;
}

double Tree::info(double yes, double no)
{
    double sum = yes + no;
    yes = yes <= 0 ? 0.001 : yes;
    no  = no  <= 0 ? 0.001 : no;
    double info = -yes / sum * log(yes / sum)
                  -no  / sum * log(no  / sum);
    return info;
}

double Tree::info(double attr[][2], double total, int option_count)
{
    double info = 0;
    for (int i = 0; i < option_count; i++)
    {
        double sum = attr[i][0] + attr[i][1];
        info += sum / total * Tree::info(attr[i][0], attr[i][1]);
    }
    return info;
}

double Tree::gain(double attr[][2], int option_count)
{
    double yes = 0;
    double no  = 0;
    for (int i = 0; i < option_count; i++)
    {
        yes += attr[i][0];
        no  += attr[i][1];
    }
    return Tree::info(yes, no) - Tree::info(attr, yes + no, option_count);
}

double Tree::splitInfo(double attr[][2], double total, int option_count)
{
    double info = 0;
    for (int i = 0; i < option_count; i++)
    {
        double sum = attr[i][0] + attr[i][1];
        info -= sum / total * log(sum / total);
    }
    return info;
}

double Tree::gainRatio(double attr[][2], int option_count)
{
    return Tree::gain(attr, option_count) / Tree::splitInfo(attr, Tree::total(attr, option_count), option_count);
}

Column* Tree::generateColumn(int column_count, int option_min, int option_max)
{
    Column *columns = new Column[column_count];
    for (int i = 0; i < column_count; i++)
    {
        columns[i].id = i;
        columns[i].option_count = rand() % (option_max - option_min + 1) + option_min;
        columns[i].options = new Option[columns[i].option_count];
        for (int j = 0; j < columns[i].option_count; j++)
        {
            columns[i].options[j].id = j;
            columns[i].options[j].yes_count = 0;
            columns[i].options[j].no_count = 0;
            columns[i].options[j].column = &columns[i];
        }
    }
    return columns;
}

Transaction* Tree::generateTransaction(Column* columns, int column_count, int transaction_count)
{
    Transaction *transactions = new Transaction[transaction_count];
    for (int i = 0; i < transaction_count; i++)
    {
        transactions[i].attribute_count = column_count;
        transactions[i].attributes = new int[column_count];
        for (int j = 0; j < column_count; j++)
        {
            transactions[i].attributes[j] = rand() % columns[j].option_count;
        }
        transactions[i].result = rand() % 2;
    }
    return transactions;
}

void Tree::scan(Node* node, Transaction* transactions, int transaction_count, Column* columns, int column_count, int gpu)
{
	// Reset the yes and no counters for all the column options
    for (int i = 0; i < column_count; i++)
    {
        for (int j = 0; j < columns[i].option_count; j++)
        {
            columns[i].options[j].yes_count = 0;
            columns[i].options[j].no_count = 0;
        }
    }
	/* ----------------------------------------------------------------------------------------------------*/
	/* Count the valid transcations number in each option of each column.                                  */
    /* This block takes 99% of cpu run time. It is should be easy to be optimized using map reduce in GPU. */
	/* ----------------------------------------------------------------------------------------------------*/
    
    int valid_count = 0;
    if (gpu == 0)
    {
        valid_count = update(node, transactions, transaction_count, columns, column_count);
    }
    else if (gpu == 1)
    {
        valid_count = update_gpu(node, transactions, transaction_count, columns, column_count, 256);
    }
    else if (gpu == 2)
    {
        valid_count = update_gpu_2(node, transactions, transaction_count, columns, column_count, 256);
    }

 //   int valid_count = 0;
	//// Check each transaction if it belongs to the current node.
 //   for (int i = 0; i < transaction_count; i++)
 //   {
 //       // flag = true:  transaction is valid. (default)
 //       // flag = false: transaction is not valid.
 //       bool flag = true;
 //       Node* current = node;
	//	// Skip root node since it does not have an option, which means all the transactions are valid.
 //       while (current->option != NULL)
 //       {
 //           // Check if the current node's option id is equal to the transaction's attribute id in the given column.
 //           if (transactions[i].attributes[current->option->column->id] != current->option->id)
 //           {
 //               flag = false;
 //               break;
 //           }
 //           else
 //           {
 //               // If the transaction is valid for the current node, continue to check the parent node.
 //               current = current->parent;
 //           }
 //       }
 //       // If the transaction is valid, compute the yes and no counters for each remaining column
 //       if (flag)
 //       {
 //           valid_count++;
 //           for (int j = 0; j < column_count; j++)
 //           {
 //               if (transactions[i].result == 1)
 //               {
 //                   columns[j].options[transactions[i].attributes[j]].yes_count++;
 //               }
 //               else
 //               {
 //                   columns[j].options[transactions[i].attributes[j]].no_count++;
 //               }
 //           }
 //       }
 //   }
	/* ----------------------------------------------------------------------------------------------------*/
    /* End of the GPU optimization code block                                                              */
	/* ----------------------------------------------------------------------------------------------------*/

    node->count = valid_count;
    // Return if valid transaction is less than 10 for the current node
    if (valid_count < 10)
    {
        return;
    }
    // Initialize the information gain ratio for the current node to a minimum value.
    // So that it less than any real possible computed gain ratio value.
	node->gain_ratio = -1000000.0;
    // Travel through each column to decide the best split column for this node
    for (int i = 0; i < column_count; i++)
    {
        // flag = true:  column is valid. (default)
        // flag = false: column is not valid.
        bool flag = true;
        Node* current = node->parent;
        while (current != NULL)
        {
            if (current->column->id == columns[i].id)
            {
                flag = false;
                break;
            }
            else
            {
                // If the column is valid for the current node, continue to check the parent node.
                current = current->parent;
            }
        }
        // If the column is valid, compute the gain ratio and update the best split column information
        if (flag)
        {
            double gain = Tree::gainRatio(columns + i);
            // Update the split column for the current node if find a better gain ratio
            if (gain > node->gain_ratio)
            {
                node->gain_ratio = gain;
                node->column = columns + i;
            }
        }
    }
}

void Tree::build(Node* root, Transaction* transactions, int transaction_count , Column* columns, int column_count, int gpu)
{
	// Compute depth
	// Travel all the ancestor nodes in the current path, return if all the columns are in the path already.
    int count = 0;
    Node* current = root->parent;
    while (current != NULL)
    {
        count++;
        current = current->parent;
    }
    if (count >= column_count)
    {
        return;
    }
	// End computing depth
    else
    {
		// Compute the information gain for all the remaining columns and chose the best one for the current node
        Tree::scan(root, transactions, transaction_count, columns, column_count, gpu);
        // Return if no split column selected for the current node
        if (root->column == NULL)
        {
            return;
        }
        // Create children with the number of option count of the current split column
        int optionCount = root->column->option_count;
        root->children_count = optionCount;
        root->children = new Node[optionCount];
        for (int i = 0; i < optionCount; i++)
        {
            // Connect the children node into the the decision tree
            Node* child = root->children + i;
            Tree::iniNode(child);
            child->parent = root;
            // Set the option id for each children node
            child->option = root->column->options + i;
            // Recursively build the children nodes
            Tree::build(child, transactions, transaction_count, columns, column_count, gpu);
        }
    }
}

double Tree::gainRatio(Column* column)
{
    double attr[MAX_OPTION_COUNT][2];
    for (int i = 0; i < column->option_count; i++)
    {
        attr[i][0] = column->options[i].yes_count;
        attr[i][1] = column->options[i].no_count;
    }
    return Tree::gainRatio(attr, column->option_count);
}

void Tree::iniNode(Node* node)
{
    node->children = NULL;
    node->children_count = 0;
    node->column = NULL;
    node->count = 0;
    node->gain_ratio = 0;
    node->option = NULL;
    node->parent = NULL;
}

void Tree::DFSTravel(Node* node)
{
    if (node == NULL)
    {
        return;
    }
    else
    {
        Tree::visit(node);
        if (node->children != NULL)
        {
            for (int i = 0; i < node->children_count; i++)
            {
                Tree::DFSTravel(node->children + i);
            }
        }
    }
}

void Tree::visit(Node* node)
{
    if (node->parent != NULL)
    {
        std::cout << "column " << node->parent->column->id << "\t";
    }
    else
    {
        std::cout << "null    \t";
    }
    if (node->option != NULL)
    {
        std::cout << "option " << node->option->id << "\t";
    }
    else
    {
        std::cout << "null    \t";
    }
    if (node->column != NULL)
    {
        std::cout << "column " << node->column->id << "\t";
    }
    else
    {
        std::cout << "null    \t";
    }
    if (node->option != NULL)
    {
        std::cout << "G: " << node->gain_ratio << "\t";
    }
    else
    {
        std::cout << "null    \t\t";
    }
    std::cout << "count: " << node->count << std::endl;
}

using namespace std;

void Tree::readTable(char path[])
{
    vector<Column> columns;
    vector<Transaction> transactions;
    ifstream file;
    file.open(path);
    if (file.is_open())
    {
        string line;
        getline(file, line);
        vector<char*> tokens = Tree::split((char*) line.c_str(), "\t");
        int columnCount = tokens.size();
        for (int i = 0; i < columnCount; i++)
        {
            int columnId = atoi(Tree::split(Tree::split(tokens[i], " ")[1], "(")[0]);
            int optionCount = atoi(Tree::split(Tree::split(Tree::split(tokens[i], " ")[1], "(")[1], ")")[0]);
            Column* column = new Column;
            column->id = columnId;
            column->option_count = optionCount;
            column->options = new Option[optionCount];
            for (int j = 0; j < optionCount; j++)
            {
                column->options[j].column = column;
                column->options[j].id = j;
                column->options[j].no_count = 0;
                column->options[j].yes_count = 0;
            }
            columns.push_back(*column);
        }
        while (file.good())
        {
            getline(file, line);
            if (line.size() > 0)
            {
                Transaction transaction;
                transaction.attribute_count = columnCount;
                transaction.attributes = new int[columnCount];
                tokens = Tree::split((char*) line.c_str(), "\t");
                for (int i = 0; i < columnCount; i++)
                {
                    transaction.attributes[i] = atoi(tokens[i]);
                }
                transaction.result = atoi(tokens[columnCount]);
                // breakpoint
                //if (transactions.size() > 998)
                //{
                //    int loc = 1;
                //}
                // end breakpoint
                transactions.push_back(transaction);
            }
        }
        this->transactionCount = transactions.size();
        this->transactions = new Transaction[transactions.size()];
        this->columnCount = columns.size();
        this->columns = new Column[columns.size()];
        for (unsigned int i = 0; i < transactions.size(); i++)
        {
            this->transactions[i] = transactions[i];
        }
        for (unsigned int i = 0; i < columns.size(); i++)
        {
            this->columns[i] = columns[i];
        }
        file.close();
    }
}


vector<char*> Tree::split(char* str, char* delim)
{
    char* loc_str = new char[strlen(str)];
    strcpy(loc_str, str);
    vector<char*> tokens;
    char* token = strtok(loc_str, delim);
    while (token != NULL)
    {
        tokens.push_back(token);
        token = strtok(NULL, delim);
    }
    return tokens;
}
