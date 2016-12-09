#include <stdio.h>
#include <stdlib.h>

#include "../include/queue.h"

void plpo_queue_init(PLPOQueue_t* q) {
    q->length = 0;
    q->head = NULL;
    q->tail = NULL;
}

void plpo_queue_enqueue(PLPOQueue_t* q, PLPONode_t* ptr) {
    ptr->next = NULL;
    ptr->prev = q->tail;
    
    if (q->tail == NULL) {
        q->head = ptr;
    } else {
        q->tail->next = ptr;
    }
    
    q->tail = ptr;
    q->length++;
}

PLPONode_t* plpo_queue_dequeue(PLPOQueue_t* q) {
    if (q->head == NULL) return NULL;

    PLPONode_t* ptr = q->head;
    q->head = ptr->next;
    if (q->head == NULL) {
        q->tail = NULL;
    }
    q->length--;

    return ptr;
}

char* plpo_queue_peek(PLPOQueue_t* q) {
    if (q->head == NULL) {
        return NULL;
    } else {
        return q->head->item;
    }
}

void plpo_ringc_create(PLPORing_t* r, int length) {
    r->length = length;
    r->items = malloc(sizeof(PLPORing_Item_t*) * r->length);
    r->head = 0;
    r->tail = -1;

    PLPORing_Item_t* item;
    for (int i = 0; i < length; ++i) {
        item = &r->items[i];
        item->ptr = NULL;
        item->active = false;
    }
}

void plpo_ringc_destroy(PLPORing_t* r) {
    free(r->items);
    free(r);
    r = NULL;
}

int ring_next(int i, int max) { return (i + 1) % max; }

void plpo_ringc_put(PLPORing_t* r, void* ptr) {
    r->tail = ring_next(r->tail, r->length);
    r->items[r->tail].ptr = ptr;
    r->items[r->tail].active = true;
}

void* plpo_ringc_pop(PLPORing_t* r) {
    if (r->tail == -1) return NULL;
    PLPORing_Item_t* item = &r->items[r->head];
    if (item->active == false) return NULL;
    item->active = false;
    r->head = ring_next(r->head, r->length);

    return item->ptr;
}
