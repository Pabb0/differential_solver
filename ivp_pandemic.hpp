#ifndef tws_ivp_hpp 
#define tws_ivp_hpp

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/assignment.hpp>

namespace ublas = boost::numeric::ublas;

namespace tws {
    // Functor that calculates the system of ODEs of the SIQRD model
    template <typename T>
    struct ivp_pandemic_functor {

        typedef ublas::matrix<T> matrix_type;

        ivp_pandemic_functor( ublas::vector<T> const & params ) {
            set_params( params );
        }

        ~ivp_pandemic_functor() {}

        // Updates the parameters with the given parameters.
        void set_params( ublas::vector<T> const & params ) { 
            #ifndef NDEBUG
                assert( params.size() == 5 );
            #endif
            // params = (beta, mu, gamma, alpha, delta)
            beta_   = params( 0 );
            mu_     = params( 1 );
            gamma_  = params( 2 );
            alpha_  = params( 3 );
            delta_  = params( 4 );
        }

        // Calculates the 'f(y)' of the ODE given the state vector y
        // and stores it in the result vector
        void operator() ( ublas::vector<T> const & y, ublas::vector<T> & result ) const {
            #ifndef NDEBUG
                assert( y.size() == 5 );
                assert( y.size() == result.size() );
            #endif

            // Definition of the ODE.
            result( 0 ) = ( -beta_ * y( 1 ) * y( 0 ) ) / ( y( 0 ) + y( 1 ) + y( 3 ) ) + mu_ * y( 3 );
            result( 1 ) = ( ( beta_ * y( 0 ) ) / ( y( 0 ) + y( 1)  + y( 3)  ) - gamma_ - delta_ - alpha_ ) * y( 1 );
            result( 2 ) = delta_ * y( 1 ) - ( gamma_ + alpha_ ) * y( 2 );
            result( 3 ) = gamma_ * ( y( 1 ) + y( 2 ) ) - mu_ * y( 3 );
            result( 4 ) = alpha_ * ( y( 1 ) + y( 2 ) );
        }

        // Calculates the Jacobian of the ODE given the state vector y
        // and stores it in the result matrix
        void jacobian( ublas::vector<T> const & y, matrix_type & result ) const {
            #ifndef NDEBUG
                assert( y.size() == 5 );
                assert( result.size1() == result.size2() );
                assert( y.size() == result.size1() );
            #endif

            T s_i_r = 1.0 / ( y( 0 ) + y( 1 ) + y( 3 ) ); 
            T s_i_r2 = 1.0 / std::pow( y( 0 ) + y( 1 ) + y( 3 ), 2 );

            result <<= 
                beta_ * y( 1 ) * ( -s_i_r + y( 0 ) * s_i_r2 ),
                beta_ * y( 0 ) * ( -s_i_r + y( 1 ) * s_i_r2 ),
                0.0,
                mu_ + beta_ * y( 0 ) * y( 1 ) * s_i_r2,
                0.0,
                beta_ * y( 1 ) * ( s_i_r - y( 0 ) * s_i_r2 ),
                -( gamma_ + delta_ + alpha_ ) + beta_ * y( 0 ) * ( s_i_r - y( 1 ) * s_i_r2 ),
                0.0,
                -beta_ * y( 0 ) * y( 1 ) * s_i_r2,
                0.0,
                0.0,
                delta_,
                -( alpha_ + gamma_ ),
                0.0, 
                0.0,
                0.0,
                gamma_,
                gamma_,
                -mu_,
                0.0,
                0.0,
                alpha_,
                alpha_,
                0.0,
                0.0;
        }

        private:
            // Parameters.
            T beta_;
            T mu_;
            T gamma_;
            T alpha_;
            T delta_;
    } ; // struct ivp_pandemic_functor
} // namespace tws

#endif