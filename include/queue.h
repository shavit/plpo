#ifndef PLPO_QUEUE_H
#define PLPO_QUEUE_H

typedef struct PLPONode {
    void* item;
} PLPONode_t;

typedef struct PLPOQueue {
    int length;
} PLPOQueue_t;

PLPONode_t* plpo_queue_enqueue(PLPOQueue_t* q);
PLPONode_t* plpo_queue_dequeue(PLPOQueue_t* q);
PLPONode_t* plpo_queue_peek(PLPOQueue_t* q);

#endif // !PLPO_QUEUE_H
