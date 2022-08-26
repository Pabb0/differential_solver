#ifndef tws_bfgs_hpp
#define tws_bfgs_hpp

#include <cassert>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/lu.hpp>

#include "linesearch.hpp"
#include "gradient.hpp"

namespace tws {

    // Performs the bfgs optimization algorithm.
    // Input:   - Initial guess [pk].
    //          - Initial Hessian matrix [Bk].
    //          - Function one wants to minimize [lse].
    //          - Tolerance of the optimization algorithm [tol].
    //          - Integer to keep track of number of iterations [k].
    // After performing bfgs, the solution is placed in [pk].
    template <typename T, typename L>
    void bfgs( ublas::vector<T> & pk, ublas::matrix<T> & Bk, L const & lse, const T tol, int & k) {
        #ifndef NDEBUG
            assert( pk.size() == Bk.size2() );
            assert( tol > 0) ;
        #endif

        // Define necessary matrices and vectors.
        ublas::vector<T> dk ( pk.size() );                      // Needed to store the direction of best descent.
        ublas::vector<T> prev_pk ( pk.size() );                 // Needed to store previous pk.
        ublas::vector<T> grad_pk ( pk.size() );                 // Needed to store the gradient at pk.
        ublas::vector<T> grad_prev_pk ( pk.size() );            // Needed to store the gradient at prev_pk.
        ublas::matrix<T> Bk_temp ( Bk.size1(), Bk.size2() );    // LU changes the matrix A --> Do LU with a temp matrix.
        ublas::vector<T> s ( pk.size() );                       // Needed to store the difference of pks.
        ublas::vector<T> y ( pk.size() );                       // Needed to store the difference of gradients.
        ublas::vector<T> Bs ( pk.size() );                      // Needed to solve matrix product Bk*s.
        ublas::permutation_matrix<T> pm1( pk.size() );          // Needed to solve Ax=b using LU.
        ublas::permutation_matrix<T> pm2( pk.size() );          // Needed to reset pm1.
        ublas::vector<T> linesearch_temp ( pk.size( ));         // Temp vector that is used in the linesearch.

        T alpha;    // Stepsize of dk
        k = 0;      // Loop counter

        tws::calculate_gradient( lse, pk, grad_pk );

        while (true) {
            #ifndef NDEBUG
                std::cout << "pk at beginning of bfgs loop: " << k << " - " << pk << std::endl;
                std::cout << "Bk at beginning of bfgs loop: " << k << " - " << pk << std::endl;
            #endif
            
            // Set prev_pk and grad_prev_pk.
            prev_pk.assign( pk );
            grad_prev_pk.assign( grad_pk );

            // Find direction dk
            grad_pk.assign( -grad_pk );
            Bk_temp.assign( Bk );
            ublas::lu_factorize( Bk_temp, pm1 );            // Solve
            ublas::lu_substitute( Bk_temp, pm1, grad_pk );  // Ax=b
            pm1.assign( pm2 );                              // Reset permutation matrix
            dk.assign( grad_pk );

            // Perform linesearch.
            alpha = tws::linesearch( lse, prev_pk, dk, grad_prev_pk, linesearch_temp );

            // Take the step
            pk.assign( prev_pk + alpha*dk );
    
            s.assign( pk - prev_pk );

            // Stopping criterion.
            if ( ublas::norm_2(s) <= tol*ublas::norm_2( prev_pk ) ) return; 

            tws::calculate_gradient( lse, pk, grad_pk );
            y.assign( grad_pk - grad_prev_pk );

            // Helper matrix to update Bk.
            Bs.assign( ublas::prod( Bk, s ) );

            // Update Bk.
            Bk.assign( Bk - ublas::outer_prod( Bs, Bs) / ublas::inner_prod( s, Bs )
                          + ublas::outer_prod( y, y ) / ublas::inner_prod( s, y ) );

            k += 1;
        } // while true
    } // void bfgs()
} // namespace tws

#endif