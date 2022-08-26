#ifndef tws_ivp_solvers_hpp
#define tws_ivp_solvers_hpp

#include <cassert>
#include <limits>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/banded.hpp>
#include <boost/numeric/ublas/lu.hpp>

namespace ublas = boost::numeric::ublas;

namespace tws {

    template <typename T, typename F>
    struct forward_euler {
        // Constructor
        forward_euler(int system_size)
        : temp_y_1_( system_size ),
          temp_y_2_( system_size )
        {}

        // Destructor
        ~forward_euler() {}

        void integrate( ublas::matrix<T> & y, ublas::vector<T> & t, ublas::vector<T> const & y0, T const & dt, F const & f ) {
            // y is the matrix that will contain the solution.
            // t is the vector that will contain the time points.
            // y0 is the vector that contains the initial solution.
            // dt contains the timestep.
            // F is the defining ivp.

            #ifndef NDEBUG
                assert( y.size1() == t.size() );
                assert( y.size2() == y0.size() );
            #endif

            // Set initial condition
            ublas::row ( y, 0 ).assign(y0);
            t( 0 ) = 0.0;

            // Perform FE step iteratively
            for ( decltype( y.size1() ) i = 1; i < y.size1(); ++i ) {

                ublas::matrix_row<ublas::matrix<T>> next_y( y, i );
                ublas::matrix_row<ublas::matrix<T>> prev_y( y, i - 1 );

                temp_y_1_.assign( prev_y );
                f( temp_y_1_, temp_y_2_ ); 
            
                next_y.assign( temp_y_1_ + dt * temp_y_2_ ); // temp_y ~= f(prev_y)
                t( i ) = t( i - 1 ) + dt;
            }

        } // forward_euler_integrate

        private:
            // Member vectors in order to reuse same memory when performing FW Euler multiple times.
            ublas::vector<T> temp_y_1_;
            ublas::vector<T> temp_y_2_;
    } ; // struct forward_euler

    template <typename T, typename F>
    struct heun {
        // Constructor
        heun( int system_size )
        : 
        temp_y_1_( system_size ),
        temp_y_2_( system_size ),
        temp_y_3_( system_size ),
        temp_y_4_( system_size )
        {}

        // Destructor
        ~heun() {}
    
        void integrate( ublas::matrix<T> & y, ublas::vector<T> & t, ublas::vector<T> const & y0, T const & dt, F const & f ) {
            // y is the matrix that will contain the solution.
            // t is the vector that will contain the time points.
            // y0 is the vector that contains the initial solution.
            // dt contains the timestep.
            // F is the defining ivp.

            #ifndef NDEBUG
                assert( y.size1() == t.size() );
                assert( y.size2() == y0.size() );
            #endif

            // Set initial condition.
            ublas::row ( y, 0 ).assign(y0);
            t( 0 ) = 0.0;
            
            // Perform heun step iteratively.
            for ( decltype( y.size1() ) i = 1; i < y.size1(); ++i ) {
                ublas::matrix_row<ublas::matrix<T>> next_y( y, i );
                ublas::matrix_row<ublas::matrix<T>> prev_y( y, i - 1 );

                temp_y_1_.assign( prev_y );
                f( temp_y_1_, temp_y_2_); 

                temp_y_3_.assign( temp_y_1_ + dt * temp_y_2_ );
                f( temp_y_3_, temp_y_4_);
            
                next_y.assign( temp_y_1_ + dt / 2 * ( temp_y_2_ + temp_y_4_ ) );
                t( i ) = t( i - 1 ) + dt;
            }

        } // heun_integrate

        private:
            // Member vectors in order to reuse same memory when performing FW Euler multiple times.
            ublas::vector<T> temp_y_1_;
            ublas::vector<T> temp_y_2_;
            ublas::vector<T> temp_y_3_;
            ublas::vector<T> temp_y_4_;
    } ; // struct heun

    template <typename T, typename F>
    struct backward_euler {
        // Constructor
        backward_euler(int system_size)
        : 
        temp_y_1_( system_size ),
        temp_y_2_( system_size ),
        temp_y_3_( system_size ),
        b_( system_size ),
        I_( system_size ),
        A_( system_size, system_size ),
        jac_matrix_( system_size, system_size ),
        pm1_( system_size ),
        pm2_( system_size )
        {}

        // Destructor
        ~backward_euler() {}

        // GENERAL FULL JACOBIAN
        template <typename U=F>
        typename std::enable_if<std::is_same<typename U::matrix_type, ublas::matrix<T> >::value>::type
        integrate( ublas::matrix<T> & y, ublas::vector<T> & t, ublas::vector<T> const & y0, T const & dt, F const & f,
        T const eps=1e4*std::numeric_limits<double>::epsilon() ) {
            // y is the matrix that will contain the solution.
            // t is the vector that will contain the time points.
            // y0 is the vector that contains the initial solution.
            // dt (number) contains the timestep.
            // f (functor) is the defining ivp.
            // f also needs to have a member function jacobian to calculate the jacobian of the ivp.
            // eps (number, optional) specifies the accuracy of the iterative algorithm used in backward euler.
            #ifndef NDEBUG
                assert( y.size1() == t.size() );
                assert( y.size2() == y0.size() );
            #endif

            // Set initial condition
            ublas::row ( y, 0 ).assign(y0);
            t( 0 ) = 0.0;
            
            for ( decltype( y.size1() ) i = 1; i < y.size1(); ++i ) {
                    
                ublas::matrix_row<ublas::matrix<T>> next_y( y, i );
                ublas::matrix_row<ublas::matrix<T>> prev_y( y, i - 1 );
                temp_y_1_.assign( prev_y );
                    
                f( temp_y_1_, temp_y_2_ ); 
                temp_y_2_.assign( temp_y_1_ + dt * temp_y_2_ );  // temp_y_2 is our inital guess (FE)

                while( true ) { // Newton solver

                    f( temp_y_2_, temp_y_3_ ); // temp_y_3 ~= f(temp_y_2)
                    b_.assign( temp_y_1_ + dt * temp_y_3_ - temp_y_2_ ); //  b = (prev_y + dt*f(y_next) - y_next) | y_next ~= temp_y_2

                    if ( ublas::norm_2(b_) / ublas::norm_2( temp_y_2_ ) < eps ) break; // Stopping criterion
                        
                    f.jacobian( temp_y_2_, jac_matrix_ );
                    A_.assign( I_ - dt*jac_matrix_ );

                    ublas::lu_factorize( A_, pm1_ );         // Solve
                    ublas::lu_substitute( A_, pm1_, b_ );    // Ax=b
                    pm1_.assign( pm2_ );                      // Reset permutation matrix

                    temp_y_2_.assign( temp_y_2_ + b_ );                        
                }

                next_y.assign( temp_y_2_ );
                t( i ) = t( i - 1 ) + dt;
            } // for ( decltype( y.size1() ) i = 1; i < y.size1(); ++i )    
        } // integrate (general matrix)


        // DIAGONAL JACOBIAN
        template<typename U=F>
        typename std::enable_if<std::is_same<typename U::matrix_type, ublas::diagonal_matrix<T> >::value>::type
        integrate( ublas::matrix<T> & y, ublas::vector<T> & t, ublas::vector<T> const & y0, T const & dt, F const & f,
        T const eps=1e4*std::numeric_limits<double>::epsilon() ) {
            // y is the matrix that will contain the solution.
            // t is the vector that will contain the time points.
            // y0 is the vector that contains the initial solution.
            // dt (number) contains the timestep.
            // f (functor) is the defining ivp.
            // f also needs to have a member function jacobian to calculate the jacobian of the ivp.
            // eps (number, optional) specifies the accuracy of the iterative algorithm used in backward euler.

            #ifndef NDEBUG
                assert( y.size1() == t.size() );
                assert( y.size2() == y0.size() );
            #endif

            // Set initial condition
            ublas::row ( y, 0 ).assign( y0 );
            t( 0 ) = 0.0;
            
            for ( decltype( y.size1() ) i = 1; i < y.size1(); ++i ) {
                ublas::matrix_row<ublas::matrix<T>> next_y( y, i );
                ublas::matrix_row<ublas::matrix<T>> prev_y( y, i - 1 );
                temp_y_1_.assign( prev_y );
                    
                f( temp_y_1_, temp_y_2_ ); 
                temp_y_2_.assign( temp_y_1_ + dt * temp_y_2_ );  // temp_y_2 is our inital guess (FE)

                while( true ) { // Newton solver

                    f( temp_y_2_, temp_y_3_ ); // temp_y_3 ~= f(temp_y_2)
                    b_.assign( temp_y_1_ + dt * temp_y_3_ - temp_y_2_ ); //  b = (prev_y + dt*f(y_next) - y_next) | y_next ~= temp_y_2
                    
                    if ( ublas::norm_2( b_ ) / ublas::norm_2( temp_y_2_ ) < eps ) break; // Stopping criterion.
                    
                    f.jacobian( temp_y_2_, jac_matrix_ );
                    A_.assign( I_ - dt*jac_matrix_ );
                    
                    // Iterate over the diagonal to solve Ax=b
                    for ( decltype( A_.size1() ) i = 0; i < A_.size1(); ++i ) b_( i ) /= A_( i, i );
                    
                    temp_y_2_.assign( temp_y_2_ + b_ );
                }

                next_y.assign( temp_y_2_ );
                t( i ) = t( i - 1 ) + dt;
            } // for ( decltype( y.size1() ) i = 1; i < y.size1(); ++i ) 
        } // integrate (diagonal matrix)


    private:
        // Member vectors/matrices in order to reuse same memory when performing BW Euler multiple times.
        ublas::vector<T> temp_y_1_;
        ublas::vector<T> temp_y_2_;
        ublas::vector<T> temp_y_3_;
        ublas::vector<T> b_;
        ublas::identity_matrix<T> I_;
        typename F::matrix_type A_;
        typename F::matrix_type jac_matrix_;
        ublas::permutation_matrix<T> pm1_;
        ublas::permutation_matrix<T> pm2_;
    } ; // struct backward_euler

} // namespace tws

#endif