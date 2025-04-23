#ifndef EdgeInfo__H__
#define EdgeInfo__H__

#include <ostream>
#include <type_traits>
struct EdgeType {
    using IntType = int;
    using FloatType = float;
    IntType to;
    FloatType weight;
};
inline std::ostream& operator<<(std::ostream& os, EdgeType edge){
    return os << edge.to << ',' << edge.weight;
}
static_assert(std::is_trivial<EdgeType>::value);

#endif
