#ifndef tws_gradient_hpp
#define tws_gradient_hpp

#include <boost/numeric/ublas/vector.hpp>
#include <limits>
#include <cassert>

namespace ublas = boost::numeric::ublas; 

namespace tws {
    // Calculates the gradient of a function f in a point pk
    // using finite differences.
    // The solution is put into grad_pk.
    template<typename F, typename T>
    void calculate_gradient( F & f, ublas::vector<T> & pk, ublas::vector<T> & grad_pk) {
        #ifndef NDEBUG
            assert ( pk.size() == grad_pk.size() );
        #endif

        T h = std::sqrt( std::numeric_limits<double>::epsilon() );
        T lse = f( pk );
        T temp;

        // Perform a finite difference for every variable.
        for ( decltype ( pk.size() ) i = 0; i < pk.size(); ++i ) {
            temp = pk( i );
            pk( i ) += h;
            grad_pk( i ) = ( f( pk ) - lse ) / h;
            pk( i ) = temp;
        }        
    } // calculte_gradient()
} // tws

#endif