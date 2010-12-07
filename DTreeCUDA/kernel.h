/*
 * @author Tong Pan
 * #date 11/28/2010
 */

#ifndef _KERNEL_H_
#define _KERNEL_H_

#define MAX_OPTION_COUNT 10
#define MAX_COLUMN_COUNT 20
#define MAX_TRANSACTION_COUNT 20000

struct KOption;
struct KColumn;
struct KNode;
struct KTransaction;

struct KOption
{
    int id;
    int column_id;
    int no_count;
    int yes_count;
};

struct KColumn
{
    int id;
    KOption options[MAX_OPTION_COUNT];
    int option_count;
};

struct KTransaction
{
    int attributes[MAX_COLUMN_COUNT];
    int attribute_count;
    int result;
};

struct KNode
{
    int column_id;
    double gain_ratio;
    int option_id;
    int count;
};

#endif