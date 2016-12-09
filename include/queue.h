#ifndef PLPO_QUEUE_H
#define PLPO_QUEUE_H

#include <stdbool.h>

typedef struct PLPONode {
    char* item;
    struct PLPONode* next;
    struct PLPONode* prev;
} PLPONode_t;

typedef struct PLPOQueue {
    int length;
    PLPONode_t* head;
    PLPONode_t* tail;
} PLPOQueue_t;

void plpo_queue_init(PLPOQueue_t* q);
void plpo_queue_destroy(PLPOQueue_t* q);
void plpo_queue_enqueue(PLPOQueue_t* q, PLPONode_t* ptr);
PLPONode_t* plpo_queue_dequeue(PLPOQueue_t* q);
char* plpo_queue_peek(PLPOQueue_t* q);

typedef struct PLPORing_Item {
    void* ptr;
    bool active;
} PLPORing_Item_t;

typedef struct PLPORing {
    int length;
    PLPORing_Item_t* items;
    int tail;
    int head;
} PLPORing_t;

void plpo_ringc_create(PLPORing_t* r, int length);
void plpo_ringc_destroy(PLPORing_t* r);
void plpo_ringc_put(PLPORing_t* r, void* ptr);
void* plpo_ringc_pop(PLPORing_t* r);
void* plpo_ringc_emit(PLPORing_t*);

#endif // !PLPO_QUEUE_H
