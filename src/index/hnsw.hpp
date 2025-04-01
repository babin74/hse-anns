#pragma once

#include <assert.h>
#include <stdlib.h>
#include <atomic>
#include <iostream>
#include <memory>
#include <queue>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "space/i16.hpp"
#include "visited_list_pool.h"

struct HierarchicalNSW {
    using space_t = space::I16;
    using dist_t = space_t::dist_t;
    using comp_t = space_t::computer_t;

   public:
    static const int MAX_LABEL_OPERATION_LOCKS = 65536;
    static const unsigned char DELETE_MARK = 0x01;

    int max_elements_{0};
    mutable std::atomic<int> cur_element_count{
        0};  // current number of elements
    int size_data_per_element_{0};
    int size_links_per_element_{0};
    mutable std::atomic<int> num_deleted_{0};  // number of deleted elements
    int M_{0};
    int maxM_{0};
    int maxM0_{0};
    int ef_construction_{0};
    int ef_{0};

    double mult_{0.0}, revSize_{0.0};
    int maxlevel_{0};
    int dist_cnt{0};

    std::unique_ptr<VisitedListPool> visited_list_pool_{nullptr};

    // Locks operations with element by label value
    mutable std::vector<std::mutex> label_op_locks_;

    std::mutex global;
    std::vector<std::mutex> link_list_locks_;

    int enterpoint_node_{0};

    int size_links_level0_{0};
    int offsetData_{0}, offsetLevel0_{0}, label_offset_{0};

    char* data_level0_memory_{nullptr};
    char** linkLists_{nullptr};
    std::vector<int> element_levels_;  // keeps level of each element

    int data_size_{0};

    mutable std::mutex label_lookup_lock;  // lock for label_lookup_
    std::unordered_map<int, int> label_lookup_;

    std::default_random_engine level_generator_;
    std::default_random_engine update_probability_generator_;

    mutable std::atomic<long> metric_distance_computations{0};
    mutable std::atomic<long> metric_hops{0};

    bool allow_replace_deleted_ =
        false;  // flag to replace deleted elements (marked as deleted) during insertions

    std::mutex deleted_elements_lock;  // lock for deleted_elements
    std::unordered_set<int>
        deleted_elements;  // contains internal ids of deleted elements

    space_t& space;

    HierarchicalNSW(space_t& space) : space(space) {}

    HierarchicalNSW(space_t& space, int max_elements, int M = 16,
                    int ef_construction = 200, int random_seed = 100,
                    bool allow_replace_deleted = false)
        : label_op_locks_(MAX_LABEL_OPERATION_LOCKS),
          link_list_locks_(max_elements),
          element_levels_(max_elements),
          allow_replace_deleted_(allow_replace_deleted),
          space(space) {
        max_elements_ = max_elements;
        num_deleted_ = 0;
        data_size_ = space.Size();
        if (M <= 10000) {
            M_ = M;
        } else {
            M_ = 10000;
        }
        maxM_ = M_;
        maxM0_ = M_ * 2;
        ef_construction_ = std::max(ef_construction, M_);
        ef_ = 10;

        level_generator_.seed(random_seed);
        update_probability_generator_.seed(random_seed + 1);

        size_links_level0_ = maxM0_ * sizeof(int) + sizeof(int);
        size_data_per_element_ =
            size_links_level0_ + data_size_ + sizeof(int);
        offsetData_ = size_links_level0_;
        label_offset_ = size_links_level0_ + data_size_;
        offsetLevel0_ = 0;

        data_level0_memory_ =
            (char*)malloc(max_elements_ * size_data_per_element_);
        if (data_level0_memory_ == nullptr)
            throw std::runtime_error("Not enough memory");

        cur_element_count = 0;

        visited_list_pool_ = std::unique_ptr<VisitedListPool>(
            new VisitedListPool(1, max_elements));

        // initializations for special treatment of the first node
        enterpoint_node_ = -1;
        maxlevel_ = -1;

        linkLists_ = (char**)malloc(sizeof(void*) * max_elements_);
        if (linkLists_ == nullptr)
            throw std::runtime_error(
                "Not enough memory: HierarchicalNSW failed to allocate "
                "linklists");
        size_links_per_element_ = maxM_ * sizeof(int) + sizeof(int);
        mult_ = 1 / log(1.0 * M_);
        revSize_ = 1.0 / mult_;
    }

    ~HierarchicalNSW() { clear(); }

    void clear() {
        free(data_level0_memory_);
        data_level0_memory_ = nullptr;
        for (int i = 0; i < cur_element_count; i++) {
            if (element_levels_[i] > 0)
                free(linkLists_[i]);
        }
        free(linkLists_);
        linkLists_ = nullptr;
        cur_element_count = 0;
        visited_list_pool_.reset(nullptr);
    }

    struct CompareByFirst {
        constexpr bool operator()(
            std::pair<dist_t, int> const& a,
            std::pair<dist_t, int> const& b) const noexcept {
            return a.first < b.first;
        }
    };

    void setEf(int ef) { ef_ = ef; }

    inline std::mutex& getLabelOpMutex(int label) const {
        // calculate hash
        int lock_id = label & (MAX_LABEL_OPERATION_LOCKS - 1);
        return label_op_locks_[lock_id];
    }

    inline int getExternalLabel(int internal_id) const {
        int return_label;
        memcpy(&return_label,
               (data_level0_memory_ + internal_id * size_data_per_element_ +
                label_offset_),
               sizeof(int));
        return return_label;
    }

    inline void setExternalLabel(int internal_id, int label) const {
        memcpy((data_level0_memory_ + internal_id * size_data_per_element_ +
                label_offset_),
               &label, sizeof(int));
    }

    inline int* getExternalLabeLp(int internal_id) const {
        return (int*)(data_level0_memory_ +
                         internal_id * size_data_per_element_ + label_offset_);
    }

    inline char* getDataByInternalId(int internal_id) const {
        return (data_level0_memory_ + internal_id * size_data_per_element_ +
                offsetData_);
    }

    int getRandomLevel(double reverse_size) {
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        double r = -log(distribution(level_generator_)) * reverse_size;
        return (int)r;
    }

    int getMaxElements() { return max_elements_; }

    int getCurrentElementCount() { return cur_element_count; }

    int getDeletedCount() { return num_deleted_; }

    std::priority_queue<std::pair<dist_t, int>,
                        std::vector<std::pair<dist_t, int>>, CompareByFirst>
    searchBaseLayer(int ep_id, space_t::point_t data_point, int layer) {
        VisitedList* vl = visited_list_pool_->getFreeVisitedList();
        int* visited_array = vl->mass;
        int visited_array_tag = vl->curV;

        std::priority_queue<std::pair<dist_t, int>,
                            std::vector<std::pair<dist_t, int>>,
                            CompareByFirst>
            top_candidates;
        std::priority_queue<std::pair<dist_t, int>,
                            std::vector<std::pair<dist_t, int>>,
                            CompareByFirst>
            candidateSet;

        dist_t lowerBound;
        auto comp = space.GetComputer(data_point);
        if (!isMarkedDeleted(ep_id)) {
            dist_t dist = comp.Distance(ep_id);
            ++dist_cnt;
            top_candidates.emplace(dist, ep_id);
            lowerBound = dist;
            candidateSet.emplace(-dist, ep_id);
        } else {
            lowerBound = std::numeric_limits<dist_t>::max();
            candidateSet.emplace(-lowerBound, ep_id);
        }
        visited_array[ep_id] = visited_array_tag;

        while (!candidateSet.empty()) {
            std::pair<dist_t, int> curr_el_pair = candidateSet.top();
            if ((-curr_el_pair.first) > lowerBound &&
                top_candidates.size() == ef_construction_) {
                break;
            }
            candidateSet.pop();

            int curNodeNum = curr_el_pair.second;

            std::unique_lock<std::mutex> lock(link_list_locks_[curNodeNum]);

            int*
                data;  // = (int *)(linkList0_ + curNodeNum * size_links_per_element0_);
            if (layer == 0) {
                data = (int*)get_linklist0(curNodeNum);
            } else {
                data = (int*)get_linklist(curNodeNum, layer);
                //                    data = (int *) (linkLists_[curNodeNum] + (layer - 1) * size_links_per_element_);
            }
            int size = getListCount((int*)data);
            int* datal = (int*)(data + 1);
            for (int j = 0; j < size; j++) {
                int candidate_id = *(datal + j);
                if (visited_array[candidate_id] == visited_array_tag)
                    continue;
                visited_array[candidate_id] = visited_array_tag;
                dist_t dist1 = comp.Distance(candidate_id);
                if (top_candidates.size() < ef_construction_ ||
                    lowerBound > dist1) {
                    candidateSet.emplace(-dist1, candidate_id);
                    if (!isMarkedDeleted(candidate_id))
                        top_candidates.emplace(dist1, candidate_id);

                    if (top_candidates.size() > ef_construction_)
                        top_candidates.pop();

                    if (!top_candidates.empty())
                        lowerBound = top_candidates.top().first;
                }
            }
        }
        visited_list_pool_->releaseVisitedList(vl);

        return top_candidates;
    }

    // bare_bone_search means there is no check for deletions and stop condition is ignored in return of extra performance
    template <bool bare_bone_search = true, bool collect_metrics = false>
    std::priority_queue<std::pair<dist_t, int>,
                        std::vector<std::pair<dist_t, int>>, CompareByFirst>
    searchBaseLayerST(int ep_id, space_t::point_t data_point,
                      int ef) const {
        VisitedList* vl = visited_list_pool_->getFreeVisitedList();
        int* visited_array = vl->mass;
        int visited_array_tag = vl->curV;

        std::priority_queue<std::pair<dist_t, int>,
                            std::vector<std::pair<dist_t, int>>,
                            CompareByFirst>
            top_candidates;
        std::priority_queue<std::pair<dist_t, int>,
                            std::vector<std::pair<dist_t, int>>,
                            CompareByFirst>
            candidate_set;

        dist_t lowerBound;
        comp_t comp = space.GetComputer(data_point);
        if (bare_bone_search || (!isMarkedDeleted(ep_id))) {
            dist_t dist = comp.Distance(ep_id);
            lowerBound = dist;
            top_candidates.emplace(dist, ep_id);
            candidate_set.emplace(-dist, ep_id);
        } else {
            lowerBound = std::numeric_limits<dist_t>::max();
            candidate_set.emplace(-lowerBound, ep_id);
        }

        visited_array[ep_id] = visited_array_tag;

        while (!candidate_set.empty()) {
            std::pair<dist_t, int> current_node_pair = candidate_set.top();
            dist_t candidate_dist = -current_node_pair.first;

            bool flag_stop_search;
            if (bare_bone_search) {
                flag_stop_search = candidate_dist > lowerBound;
            } else {
                flag_stop_search =
                    candidate_dist > lowerBound && top_candidates.size() == ef;
            }
            if (flag_stop_search) {
                break;
            }
            candidate_set.pop();

            int current_node_id = current_node_pair.second;
            int* data = (int*)get_linklist0(current_node_id);
            int size = getListCount((int*)data);
            //                bool cur_node_deleted = isMarkedDeleted(current_node_id);
            if (collect_metrics) {
                metric_hops++;
                metric_distance_computations += size;
            }

            for (int j = 1; j <= size; j++) {
                int candidate_id = *(data + j);
                if (!(visited_array[candidate_id] == visited_array_tag)) {
                    visited_array[candidate_id] = visited_array_tag;
                    dist_t dist = comp.Distance(candidate_id);

                    bool flag_consider_candidate;
                    flag_consider_candidate =
                        top_candidates.size() < ef || lowerBound > dist;
                    if (flag_consider_candidate) {
                        candidate_set.emplace(-dist, candidate_id);
                        if (bare_bone_search ||
                            (!isMarkedDeleted(candidate_id))) {
                            top_candidates.emplace(dist, candidate_id);
                        }

                        bool flag_remove_extra = false;
                        flag_remove_extra = top_candidates.size() > ef;
                        while (flag_remove_extra) {
                            int id = top_candidates.top().second;
                            top_candidates.pop();
                            flag_remove_extra = top_candidates.size() > ef;
                        }

                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;
                    }
                }
            }
        }

        visited_list_pool_->releaseVisitedList(vl);
        return top_candidates;
    }

    void getNeighborsByHeuristic2(
        std::priority_queue<std::pair<dist_t, int>,
                            std::vector<std::pair<dist_t, int>>,
                            CompareByFirst>& top_candidates,
        const int M) {
        if (top_candidates.size() < M) {
            return;
        }

        std::priority_queue<std::pair<dist_t, int>> queue_closest;
        std::vector<std::pair<dist_t, int>> return_list;
        while (top_candidates.size() > 0) {
            queue_closest.emplace(-top_candidates.top().first,
                                  top_candidates.top().second);
            top_candidates.pop();
        }

        while (queue_closest.size()) {
            if (return_list.size() >= M)
                break;
            std::pair<dist_t, int> curent_pair = queue_closest.top();
            dist_t dist_to_query = -curent_pair.first;
            queue_closest.pop();
            bool good = true;

            for (std::pair<dist_t, int> second_pair : return_list) {
                dist_t curdist = space.GetComputer(second_pair.second)
                                     .Distance(curent_pair.second);
                if (curdist < dist_to_query) {
                    good = false;
                    break;
                }
            }
            if (good) {
                return_list.push_back(curent_pair);
            }
        }

        for (std::pair<dist_t, int> curent_pair : return_list) {
            top_candidates.emplace(-curent_pair.first, curent_pair.second);
        }
    }

    int* get_linklist0(int internal_id) const {
        return (int*)(data_level0_memory_ +
                         internal_id * size_data_per_element_ + offsetLevel0_);
    }

    int* get_linklist0(int internal_id, char* data_level0_memory_) const {
        return (int*)(data_level0_memory_ +
                         internal_id * size_data_per_element_ + offsetLevel0_);
    }

    int* get_linklist(int internal_id, int level) const {
        return (int*)(linkLists_[internal_id] +
                         (level - 1) * size_links_per_element_);
    }

    int* get_linklist_at_level(int internal_id, int level) const {
        return level == 0 ? get_linklist0(internal_id)
                          : get_linklist(internal_id, level);
    }

    int mutuallyConnectNewElement(
        const void* data_point, int cur_c,
        std::priority_queue<std::pair<dist_t, int>,
                            std::vector<std::pair<dist_t, int>>,
                            CompareByFirst>& top_candidates,
        int level, bool isUpdate) {
        int Mcurmax = level ? maxM_ : maxM0_;
        getNeighborsByHeuristic2(top_candidates, M_);
        if (top_candidates.size() > M_)
            throw std::runtime_error(
                "Should be not be more than M_ candidates returned by the "
                "heuristic");

        std::vector<int> selectedNeighbors;
        selectedNeighbors.reserve(M_);
        while (top_candidates.size() > 0) {
            selectedNeighbors.push_back(top_candidates.top().second);
            top_candidates.pop();
        }

        int next_closest_entry_point = selectedNeighbors.back();

        {
            // lock only during the update
            // because during the addition the lock for cur_c is already acquired
            std::unique_lock<std::mutex> lock(link_list_locks_[cur_c],
                                              std::defer_lock);
            if (isUpdate) {
                lock.lock();
            }
            int* ll_cur;
            if (level == 0)
                ll_cur = get_linklist0(cur_c);
            else
                ll_cur = get_linklist(cur_c, level);

            if (*ll_cur && !isUpdate) {
                throw std::runtime_error(
                    "The newly inserted element should have blank link list");
            }
            setListCount(ll_cur, selectedNeighbors.size());
            int* data = (int*)(ll_cur + 1);
            for (int idx = 0; idx < selectedNeighbors.size(); idx++) {
                if (data[idx] && !isUpdate)
                    throw std::runtime_error("Possible memory corruption");
                if (level > element_levels_[selectedNeighbors[idx]])
                    throw std::runtime_error(
                        "Trying to make a link on a non-existent level");

                data[idx] = selectedNeighbors[idx];
            }
        }

        for (int idx = 0; idx < selectedNeighbors.size(); idx++) {
            std::unique_lock<std::mutex> lock(
                link_list_locks_[selectedNeighbors[idx]]);

            int* ll_other;
            if (level == 0)
                ll_other = get_linklist0(selectedNeighbors[idx]);
            else
                ll_other = get_linklist(selectedNeighbors[idx], level);

            int sz_link_list_other = getListCount(ll_other);

            if (sz_link_list_other > Mcurmax)
                throw std::runtime_error("Bad value of sz_link_list_other");
            if (selectedNeighbors[idx] == cur_c)
                throw std::runtime_error(
                    "Trying to connect an element to itself");
            if (level > element_levels_[selectedNeighbors[idx]])
                throw std::runtime_error(
                    "Trying to make a link on a non-existent level");

            int* data = (int*)(ll_other + 1);

            bool is_cur_c_present = false;
            if (isUpdate) {
                for (int j = 0; j < sz_link_list_other; j++) {
                    if (data[j] == cur_c) {
                        is_cur_c_present = true;
                        break;
                    }
                }
            }

            // If cur_c is already present in the neighboring connections of `selectedNeighbors[idx]` then no need to modify any connections or run the heuristics.
            if (!is_cur_c_present) {
                if (sz_link_list_other < Mcurmax) {
                    data[sz_link_list_other] = cur_c;
                    setListCount(ll_other, sz_link_list_other + 1);
                } else {
                    dist_t d_max = space.GetComputer(selectedNeighbors[idx])
                                       .Distance(cur_c);
                    // Heuristic:
                    std::priority_queue<std::pair<dist_t, int>,
                                        std::vector<std::pair<dist_t, int>>,
                                        CompareByFirst>
                        candidates;
                    candidates.emplace(d_max, cur_c);

                    for (int j = 0; j < sz_link_list_other; j++) {
                        candidates.emplace(
                            space.GetComputer(selectedNeighbors[idx])
                                .Distance(data[j]),
                            data[j]);
                    }

                    getNeighborsByHeuristic2(candidates, Mcurmax);

                    int indx = 0;
                    while (candidates.size() > 0) {
                        data[indx] = candidates.top().second;
                        candidates.pop();
                        indx++;
                    }

                    setListCount(ll_other, indx);
                    // Nearest K:
                    /*int indx = -1;
                    for (int j = 0; j < sz_link_list_other; j++) {
                        dist_t d = fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(rez[idx]), dist_func_param_);
                        if (d > d_max) {
                            indx = j;
                            d_max = d;
                        }
                    }
                    if (indx >= 0) {
                        data[indx] = cur_c;
                    } */
                }
            }
        }

        return next_closest_entry_point;
    }

    void resizeIndex(int new_max_elements) {
        if (new_max_elements < cur_element_count)
            throw std::runtime_error(
                "Cannot resize, max element is less than the current number of "
                "elements");

        visited_list_pool_.reset(new VisitedListPool(1, new_max_elements));

        element_levels_.resize(new_max_elements);

        std::vector<std::mutex>(new_max_elements).swap(link_list_locks_);

        // Reallocate base layer
        char* data_level0_memory_new = (char*)realloc(
            data_level0_memory_, new_max_elements * size_data_per_element_);
        if (data_level0_memory_new == nullptr)
            throw std::runtime_error(
                "Not enough memory: resizeIndex failed to allocate base layer");
        data_level0_memory_ = data_level0_memory_new;

        // Reallocate all other layers
        char** linkLists_new =
            (char**)realloc(linkLists_, sizeof(void*) * new_max_elements);
        if (linkLists_new == nullptr)
            throw std::runtime_error(
                "Not enough memory: resizeIndex failed to allocate other "
                "layers");
        linkLists_ = linkLists_new;

        max_elements_ = new_max_elements;
    }

    int indexFileSize() const {
        int size = 0;
        size += sizeof(offsetLevel0_);
        size += sizeof(max_elements_);
        size += sizeof(cur_element_count);
        size += sizeof(size_data_per_element_);
        size += sizeof(label_offset_);
        size += sizeof(offsetData_);
        size += sizeof(maxlevel_);
        size += sizeof(enterpoint_node_);
        size += sizeof(maxM_);

        size += sizeof(maxM0_);
        size += sizeof(M_);
        size += sizeof(mult_);
        size += sizeof(ef_construction_);

        size += cur_element_count * size_data_per_element_;

        for (int i = 0; i < cur_element_count; i++) {
            unsigned int linkListSize =
                element_levels_[i] > 0
                    ? size_links_per_element_ * element_levels_[i]
                    : 0;
            size += sizeof(linkListSize);
            size += linkListSize;
        }
        return size;
    }

    template <typename data_t>
    std::vector<data_t> getDataByLabel(int label) const {
        // lock all operations with element by label
        std::unique_lock<std::mutex> lock_label(getLabelOpMutex(label));

        std::unique_lock<std::mutex> lock_table(label_lookup_lock);
        auto search = label_lookup_.find(label);
        if (search == label_lookup_.end() || isMarkedDeleted(search->second)) {
            throw std::runtime_error("Label not found");
        }
        int internalId = search->second;
        lock_table.unlock();

        char* data_ptrv = getDataByInternalId(internalId);
        int dim = space.dim;
        std::vector<data_t> data;
        data_t* data_ptr = (data_t*)data_ptrv;
        for (int i = 0; i < dim; i++) {
            data.push_back(*data_ptr);
            data_ptr += 1;
        }
        return data;
    }

    /*
    * Marks an element with the given label deleted, does NOT really change the current graph.
    */
    void markDelete(int label) {
        // lock all operations with element by label
        std::unique_lock<std::mutex> lock_label(getLabelOpMutex(label));

        std::unique_lock<std::mutex> lock_table(label_lookup_lock);
        auto search = label_lookup_.find(label);
        if (search == label_lookup_.end()) {
            throw std::runtime_error("Label not found");
        }
        int internalId = search->second;
        lock_table.unlock();

        markDeletedInternal(internalId);
    }

    /*
    * Uses the last 16 bits of the memory for the linked list size to store the mark,
    * whereas maxM0_ has to be limited to the lower 16 bits, however, still large enough in almost all cases.
    */
    void markDeletedInternal(int internalId) {
        assert(internalId < cur_element_count);
        if (!isMarkedDeleted(internalId)) {
            unsigned char* ll_cur =
                ((unsigned char*)get_linklist0(internalId)) + 2;
            *ll_cur |= DELETE_MARK;
            num_deleted_ += 1;
            if (allow_replace_deleted_) {
                std::unique_lock<std::mutex> lock_deleted_elements(
                    deleted_elements_lock);
                deleted_elements.insert(internalId);
            }
        } else {
            throw std::runtime_error(
                "The requested to delete element is already deleted");
        }
    }

    /*
    * Removes the deleted mark of the node, does NOT really change the current graph.
    * 
    * Note: the method is not safe to use when replacement of deleted elements is enabled,
    *  because elements marked as deleted can be completely removed by addPoint
    */
    void unmarkDelete(int label) {
        // lock all operations with element by label
        std::unique_lock<std::mutex> lock_label(getLabelOpMutex(label));

        std::unique_lock<std::mutex> lock_table(label_lookup_lock);
        auto search = label_lookup_.find(label);
        if (search == label_lookup_.end()) {
            throw std::runtime_error("Label not found");
        }
        int internalId = search->second;
        lock_table.unlock();

        unmarkDeletedInternal(internalId);
    }

    /*
    * Remove the deleted mark of the node.
    */
    void unmarkDeletedInternal(int internalId) {
        assert(internalId < cur_element_count);
        if (isMarkedDeleted(internalId)) {
            unsigned char* ll_cur =
                ((unsigned char*)get_linklist0(internalId)) + 2;
            *ll_cur &= ~DELETE_MARK;
            num_deleted_ -= 1;
            if (allow_replace_deleted_) {
                std::unique_lock<std::mutex> lock_deleted_elements(
                    deleted_elements_lock);
                deleted_elements.erase(internalId);
            }
        } else {
            throw std::runtime_error(
                "The requested to undelete element is not deleted");
        }
    }

    /*
    * Checks the first 16 bits of the memory to see if the element is marked deleted.
    */
    bool isMarkedDeleted(int internalId) const {
        unsigned char* ll_cur = ((unsigned char*)get_linklist0(internalId)) + 2;
        return *ll_cur & DELETE_MARK;
    }

    unsigned short int getListCount(int* ptr) const {
        return *((unsigned short int*)ptr);
    }

    void setListCount(int* ptr, unsigned short int size) const {
        *((unsigned short int*)(ptr)) = *((unsigned short int*)&size);
    }

    void Build() {
        for (int i = 0; i < space.Size(); ++i)
            addPoint(space.GetPoint(i), i);
    }

    /*
    * Adds point. Updates the point if it is already in the index.
    * If replacement of deleted elements is enabled: replaces previously deleted point if any, updating it with new point
    */
    void addPoint(space_t::point_t data_point, int label,
                  bool replace_deleted = false) {
        if ((allow_replace_deleted_ == false) && (replace_deleted == true)) {
            throw std::runtime_error(
                "Replacement of deleted elements is disabled in constructor");
        }

        // lock all operations with element by label
        std::unique_lock<std::mutex> lock_label(getLabelOpMutex(label));
        if (!replace_deleted) {
            addPoint(data_point, label, -1);
            return;
        }
        // check if there is vacant place
        int internal_id_replaced;
        std::unique_lock<std::mutex> lock_deleted_elements(
            deleted_elements_lock);
        bool is_vacant_place = !deleted_elements.empty();
        if (is_vacant_place) {
            internal_id_replaced = *deleted_elements.begin();
            deleted_elements.erase(internal_id_replaced);
        }
        lock_deleted_elements.unlock();

        // if there is no vacant place then add or update point
        // else add point to vacant place
        if (!is_vacant_place) {
            addPoint(data_point, label, -1);
        } else {
            // we assume that there are no concurrent operations on deleted element
            int label_replaced = getExternalLabel(internal_id_replaced);
            setExternalLabel(internal_id_replaced, label);

            std::unique_lock<std::mutex> lock_table(label_lookup_lock);
            label_lookup_.erase(label_replaced);
            label_lookup_[label] = internal_id_replaced;
            lock_table.unlock();

            unmarkDeletedInternal(internal_id_replaced);
            updatePoint(data_point, internal_id_replaced, 1.0);
        }
    }

    void updatePoint(space_t::point_t dataPoint, int internalId,
                     float updateNeighborProbability) {
        // update the feature vector associated with existing point with new vector
        memcpy(getDataByInternalId(internalId), dataPoint, data_size_);

        int maxLevelCopy = maxlevel_;
        int entryPointCopy = enterpoint_node_;
        // If point to be updated is entry point and graph just contains single element then just return.
        if (entryPointCopy == internalId && cur_element_count == 1)
            return;

        int elemLevel = element_levels_[internalId];
        std::uniform_real_distribution<float> distribution(0.0, 1.0);
        for (int layer = 0; layer <= elemLevel; layer++) {
            std::unordered_set<int> sCand;
            std::unordered_set<int> sNeigh;
            std::vector<int> listOneHop =
                getConnectionsWithLock(internalId, layer);
            if (listOneHop.size() == 0)
                continue;

            sCand.insert(internalId);

            for (auto&& elOneHop : listOneHop) {
                sCand.insert(elOneHop);

                if (distribution(update_probability_generator_) >
                    updateNeighborProbability)
                    continue;

                sNeigh.insert(elOneHop);

                std::vector<int> listTwoHop =
                    getConnectionsWithLock(elOneHop, layer);
                for (auto&& elTwoHop : listTwoHop) {
                    sCand.insert(elTwoHop);
                }
            }

            for (auto&& neigh : sNeigh) {
                // if (neigh == internalId)
                //     continue;

                std::priority_queue<std::pair<dist_t, int>,
                                    std::vector<std::pair<dist_t, int>>,
                                    CompareByFirst>
                    candidates;
                int size =
                    sCand.find(neigh) == sCand.end()
                        ? sCand.size()
                        : sCand.size() -
                              1;  // sCand guaranteed to have size >= 1
                int elementsToKeep = std::min(ef_construction_, size);
                for (auto&& cand : sCand) {
                    if (cand == neigh)
                        continue;
                    dist_t distance = space.GetComputer(neigh).Distance(cand);
                    if (candidates.size() < elementsToKeep) {
                        candidates.emplace(distance, cand);
                    } else {
                        if (distance < candidates.top().first) {
                            candidates.pop();
                            candidates.emplace(distance, cand);
                        }
                    }
                }

                // Retrieve neighbours using heuristic and set connections.
                getNeighborsByHeuristic2(candidates,
                                         layer == 0 ? maxM0_ : maxM_);

                {
                    std::unique_lock<std::mutex> lock(link_list_locks_[neigh]);
                    int* ll_cur;
                    ll_cur = get_linklist_at_level(neigh, layer);
                    int candSize = candidates.size();
                    setListCount(ll_cur, candSize);
                    int* data = (int*)(ll_cur + 1);
                    for (int idx = 0; idx < candSize; idx++) {
                        data[idx] = candidates.top().second;
                        candidates.pop();
                    }
                }
            }
        }

        repairConnectionsForUpdate(dataPoint, entryPointCopy, internalId,
                                   elemLevel, maxLevelCopy);
    }

    void repairConnectionsForUpdate(space_t::point_t dataPoint,
                                    int entryPointInternalId,
                                    int dataPointInternalId,
                                    int dataPointLevel, int maxLevel) {
        int currObj = entryPointInternalId;
        if (dataPointLevel < maxLevel) {
            dist_t curdist = space.GetComputer(dataPoint).Distance(currObj);
            for (int level = maxLevel; level > dataPointLevel; level--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    int* data;
                    std::unique_lock<std::mutex> lock(
                        link_list_locks_[currObj]);
                    data = get_linklist_at_level(currObj, level);
                    int size = getListCount(data);
                    int* datal = (int*)(data + 1);
                    for (int i = 0; i < size; i++) {
                        int cand = datal[i];
                        dist_t d = space.GetComputer(dataPoint).Distance(cand);
                        if (d < curdist) {
                            curdist = d;
                            currObj = cand;
                            changed = true;
                        }
                    }
                }
            }
        }

        if (dataPointLevel > maxLevel)
            throw std::runtime_error(
                "Level of item to be updated cannot be bigger than max level");

        for (int level = dataPointLevel; level >= 0; level--) {
            std::priority_queue<std::pair<dist_t, int>,
                                std::vector<std::pair<dist_t, int>>,
                                CompareByFirst>
                topCandidates = searchBaseLayer(currObj, dataPoint, level);

            std::priority_queue<std::pair<dist_t, int>,
                                std::vector<std::pair<dist_t, int>>,
                                CompareByFirst>
                filteredTopCandidates;
            while (topCandidates.size() > 0) {
                if (topCandidates.top().second != dataPointInternalId)
                    filteredTopCandidates.push(topCandidates.top());

                topCandidates.pop();
            }

            // Since element_levels_ is being used to get `dataPointLevel`, there could be cases where `topCandidates` could just contains entry point itself.
            // To prevent self loops, the `topCandidates` is filtered and thus can be empty.
            if (filteredTopCandidates.size() > 0) {
                bool epDeleted = isMarkedDeleted(entryPointInternalId);
                if (epDeleted) {
                    filteredTopCandidates.emplace(
                        space.GetComputer(dataPoint).Distance(
                            entryPointInternalId),
                        entryPointInternalId);
                    if (filteredTopCandidates.size() > ef_construction_)
                        filteredTopCandidates.pop();
                }

                currObj = mutuallyConnectNewElement(
                    dataPoint, dataPointInternalId, filteredTopCandidates,
                    level, true);
            }
        }
    }

    std::vector<int> getConnectionsWithLock(int internalId, int level) {
        std::unique_lock<std::mutex> lock(link_list_locks_[internalId]);
        int* data = get_linklist_at_level(internalId, level);
        int size = getListCount(data);
        std::vector<int> result(size);
        int* ll = (int*)(data + 1);
        memcpy(result.data(), ll, size * sizeof(int));
        return result;
    }

    int addPoint(space_t::point_t data_point, int label, int level) {
        int cur_c = 0;
        {
            // Checking if the element with the same label already exists
            // if so, updating it *instead* of creating a new element.
            std::unique_lock<std::mutex> lock_table(label_lookup_lock);
            auto search = label_lookup_.find(label);
            if (search != label_lookup_.end()) {
                int existingInternalId = search->second;
                if (allow_replace_deleted_) {
                    if (isMarkedDeleted(existingInternalId)) {
                        throw std::runtime_error(
                            "Can't use addPoint to update deleted elements if "
                            "replacement of deleted elements is enabled.");
                    }
                }
                lock_table.unlock();

                if (isMarkedDeleted(existingInternalId)) {
                    unmarkDeletedInternal(existingInternalId);
                }
                updatePoint(data_point, existingInternalId, 1.0);

                return existingInternalId;
            }

            if (cur_element_count >= max_elements_) {
                throw std::runtime_error(
                    "The number of elements exceeds the specified limit");
            }

            cur_c = cur_element_count;
            cur_element_count++;
            label_lookup_[label] = cur_c;
        }

        std::unique_lock<std::mutex> lock_el(link_list_locks_[cur_c]);
        int curlevel = getRandomLevel(mult_);
        if (level > 0)
            curlevel = level;

        element_levels_[cur_c] = curlevel;

        std::unique_lock<std::mutex> templock(global);
        int maxlevelcopy = maxlevel_;
        if (curlevel <= maxlevelcopy)
            templock.unlock();
        int currObj = enterpoint_node_;
        int enterpoint_copy = enterpoint_node_;

        memset(data_level0_memory_ + cur_c * size_data_per_element_ +
                   offsetLevel0_,
               0, size_data_per_element_);

        // Initialisation of the data and label
        memcpy(getExternalLabeLp(cur_c), &label, sizeof(int));
        memcpy(getDataByInternalId(cur_c), data_point, data_size_);

        if (curlevel) {
            linkLists_[cur_c] =
                (char*)malloc(size_links_per_element_ * curlevel + 1);
            if (linkLists_[cur_c] == nullptr)
                throw std::runtime_error(
                    "Not enough memory: addPoint failed to allocate linklist");
            memset(linkLists_[cur_c], 0,
                   size_links_per_element_ * curlevel + 1);
        }

        if ((signed)currObj != -1) {
            if (curlevel < maxlevelcopy) {
                dist_t curdist =
                    space.GetComputer(data_point).Distance(currObj);
                for (int level = maxlevelcopy; level > curlevel; level--) {
                    bool changed = true;
                    while (changed) {
                        changed = false;
                        int* data;
                        std::unique_lock<std::mutex> lock(
                            link_list_locks_[currObj]);
                        data = get_linklist(currObj, level);
                        int size = getListCount(data);

                        int* datal = (int*)(data + 1);
                        for (int i = 0; i < size; i++) {
                            int cand = datal[i];
                            if (cand < 0 || cand > max_elements_)
                                throw std::runtime_error("cand error");
                            dist_t d = space.GetComputer(data_point).Distance(cand);
                            if (d < curdist) {
                                curdist = d;
                                currObj = cand;
                                changed = true;
                            }
                        }
                    }
                }
            }

            bool epDeleted = isMarkedDeleted(enterpoint_copy);
            for (int level = std::min(curlevel, maxlevelcopy); level >= 0;
                 level--) {
                if (level > maxlevelcopy || level < 0)  // possible?
                    throw std::runtime_error("Level error");

                std::priority_queue<std::pair<dist_t, int>,
                                    std::vector<std::pair<dist_t, int>>,
                                    CompareByFirst>
                    top_candidates =
                        searchBaseLayer(currObj, data_point, level);
                if (epDeleted) {
                    top_candidates.emplace(
                        space.GetComputer(data_point).Distance(enterpoint_copy),
                        enterpoint_copy);
                    if (top_candidates.size() > ef_construction_)
                        top_candidates.pop();
                }
                currObj = mutuallyConnectNewElement(
                    data_point, cur_c, top_candidates, level, false);
            }
        } else {
            // Do nothing for the first element
            enterpoint_node_ = 0;
            maxlevel_ = curlevel;
        }

        // Releasing lock for the maximum level
        if (curlevel > maxlevelcopy) {
            enterpoint_node_ = cur_c;
            maxlevel_ = curlevel;
        }
        return cur_c;
    }

    std::vector<size_t> Search(space_t::point_t query_data, int k) {
        auto q = searchKnn(query_data, k);
        std::vector<size_t> res;
        while (!q.empty()) {
            res.push_back(q.top().second);
            q.pop();
        }
        return res;
    }

    std::priority_queue<std::pair<dist_t, int>> searchKnn(
        space_t::point_t query_data, int k) const {
        std::priority_queue<std::pair<dist_t, int>> result;
        if (cur_element_count == 0)
            return result;

        int currObj = enterpoint_node_;
        dist_t curdist =
            space.GetComputer(query_data).Distance(enterpoint_node_);

        for (int level = maxlevel_; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                int* data;

                data = get_linklist(currObj, level);
                int size = getListCount(data);
                metric_hops++;
                metric_distance_computations += size;

                int* datal = (int*)(data + 1);
                for (int i = 0; i < size; i++) {
                    int cand = datal[i];
                    if (cand < 0 || cand > max_elements_)
                        throw std::runtime_error("cand error");
                    dist_t d = space.GetComputer(query_data).Distance(cand);

                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                }
            }
        }

        std::priority_queue<std::pair<dist_t, int>,
                            std::vector<std::pair<dist_t, int>>,
                            CompareByFirst>
            top_candidates;
        bool bare_bone_search = !num_deleted_;
        if (bare_bone_search) {
            top_candidates =
                searchBaseLayerST<true>(currObj, query_data, std::max(ef_, k));
        } else {
            top_candidates =
                searchBaseLayerST<false>(currObj, query_data, std::max(ef_, k));
        }

        while (top_candidates.size() > k) {
            top_candidates.pop();
        }
        while (top_candidates.size() > 0) {
            std::pair<dist_t, int> rez = top_candidates.top();
            result.push(std::pair<dist_t, int>(
                rez.first, getExternalLabel(rez.second)));
            top_candidates.pop();
        }
        return result;
    }

    std::vector<std::pair<dist_t, int>> searchStopConditionClosest(
        space_t::point_t query_data) const {
        std::vector<std::pair<dist_t, int>> result;
        if (cur_element_count == 0)
            return result;

        int currObj = enterpoint_node_;
        dist_t curdist =
            space.GetComputer(query_data).Distance(enterpoint_node_);

        for (int level = maxlevel_; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                int* data;

                data = get_linklist(currObj, level);
                int size = getListCount(data);
                metric_hops++;
                metric_distance_computations += size;

                int* datal = (int*)(data + 1);
                for (int i = 0; i < size; i++) {
                    int cand = datal[i];
                    if (cand < 0 || cand > max_elements_)
                        throw std::runtime_error("cand error");
                    dist_t d = space.GetComputer(query_data).Distance(cand);
                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                }
            }
        }

        std::priority_queue<std::pair<dist_t, int>,
                            std::vector<std::pair<dist_t, int>>,
                            CompareByFirst>
            top_candidates;
        top_candidates = searchBaseLayerST<false>(currObj, query_data, 0);

        int sz = top_candidates.size();
        result.resize(sz);
        while (!top_candidates.empty()) {
            result[--sz] = top_candidates.top();
            top_candidates.pop();
        }

        return result;
    }

    void checkIntegrity() {
        int connections_checked = 0;
        std::vector<int> inbound_connections_num(cur_element_count, 0);
        for (int i = 0; i < cur_element_count; i++) {
            for (int l = 0; l <= element_levels_[i]; l++) {
                int* ll_cur = get_linklist_at_level(i, l);
                int size = getListCount(ll_cur);
                int* data = (int*)(ll_cur + 1);
                std::unordered_set<int> s;
                for (int j = 0; j < size; j++) {
                    assert(data[j] < cur_element_count);
                    assert(data[j] != i);
                    inbound_connections_num[data[j]]++;
                    s.insert(data[j]);
                    connections_checked++;
                }
                assert(s.size() == size);
            }
        }
        if (cur_element_count > 1) {
            int min1 = inbound_connections_num[0],
                max1 = inbound_connections_num[0];
            for (int i = 0; i < cur_element_count; i++) {
                assert(inbound_connections_num[i] > 0);
                min1 = std::min(inbound_connections_num[i], min1);
                max1 = std::max(inbound_connections_num[i], max1);
            }
            std::cerr << "Min inbound: " << min1 << ", Max inbound:" << max1
                      << "\n";
        }
        std::cerr << "integrity ok, checked " << connections_checked
                  << " connections\n";
    }
};
