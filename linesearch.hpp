#ifndef tws_linesearch_hpp
#define tws_linesearch_hpp

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <cassert>
#include <limits>

namespace ublas = boost::numeric::ublas;

namespace tws {
    // Performs linesearch and returns the optimal alpha.
    template<typename F, typename T>
    T linesearch( F const & fun, ublas::vector<T> const & x0, ublas::vector<T> const & dk, ublas::vector<T> const & grad, ublas::vector<T> & temp ) {
        #ifndef NDEBUG
            assert( x0.size() == dk.size() );
            assert( grad.size() == x0.size() );
        #endif
        
        T alpha = 1.0;
        T c1 = 1e-4;
        T f0 = fun( x0 );
        
        temp.assign( x0 + alpha*dk );

        while ( fun( temp ) > f0 + c1*alpha*inner_prod( dk, grad ) && alpha > std::numeric_limits<double>::epsilon() ) {
            alpha /= 2;
            temp.assign( x0 + alpha*dk );
        }

        return alpha;
    } // linesearch
} // tws

#endif