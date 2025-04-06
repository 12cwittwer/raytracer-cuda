// bvh.h or bvh_builder.h
#pragma once

#include "hittable.h"
#include "bvh.h"
#include "aabb.h"
#include <cstdlib>   // std::rand
#include <algorithm> // std::sort

int compare_hittables(const hittable& a, const hittable& b, int axis) {
    return bounding_box(a).axis_interval(axis).min < bounding_box(b).axis_interval(axis).min;
}

int build_bvh(
    hittable* objects, int start, int end,
    bvh_node* nodes, int& node_index_out
) {
    int object_span = end - start;
    int node_index = node_index_out++;
    bvh_node& node = nodes[node_index];

    int axis = std::rand() % 3;
    auto comparator = [axis](const hittable& a, const hittable& b) {
        return compare_hittables(a, b, axis);
    };

    if (object_span == 1) {
        node.left = node.right = objects[start];
    } else if (object_span == 2) {
        if (comparator(objects[start], objects[start+1])) {
            node.left = objects[start];
            node.right = objects[start+1];
        } else {
            node.left = objects[start+1];
            node.right = objects[start];
        }
    } else {
        std::sort(objects + start, objects + end, comparator);
        int mid = start + object_span / 2;

        int left_index  = build_bvh(objects, start, mid, nodes, node_index_out);
        int right_index = build_bvh(objects, mid, end, nodes, node_index_out);

        node.left  = { hittable_type::bvh_node, &nodes[left_index] };
        node.right = { hittable_type::bvh_node, &nodes[right_index] };
    }

    node.bbox = surrounding_box(bounding_box(node.left), bounding_box(node.right));
    return node_index;
}
