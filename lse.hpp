#ifndef tws_lse_hpp
#define tws_lse_hpp

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <limits>

#include "ivp_solvers.hpp"
#include "ivp_pandemic.hpp"

namespace ublas = boost::numeric::ublas; 

namespace tws {
    template <typename T, typename Ode>
    struct lse_functor {
        lse_functor( ublas::matrix<T> const & x_obs, ublas::vector<T> const & y0, ublas::vector<T> const & p0, Ode const & ode )
        : 
        x_obs_( x_obs ),
        y_( 8 * ( x_obs.size1() - 1 ) + 1, x_obs.size2(), 0.0 ),
        t_( 8 * ( x_obs.size1() - 1 ) + 1, 0.0 ),
        y0_( y0 ),
        ivp_pandemic_functor_obj_( p0 ),
        ode_solver_( ode )
            {   
            // For this model the population remains constant, so calculate its size beforehand.
            pop_size_squared_ = 0.0;
            for ( decltype ( x_obs_.size2() ) i = 0; i < x_obs_.size2(); ++i ) pop_size_squared_ += x_obs_( 0, i );
            pop_size_squared_ = std::pow( pop_size_squared_, 2 );

            total_time_ = x_obs_.size1() - 1;
            }

        ~lse_functor() {}

        T operator() ( ublas::vector<T> const & p ) const {
            #ifndef NDEBUG 
                assert( p.size() == 5 );
                std::cout << "Parameters at LSE evaluation: " << p << std::endl;
            #endif
                
            // If one of the given parameters is negative
            // the solution is infeasible, hence infinity is returned.
            if ( std::any_of(p.begin(), p.end(), []( auto element ) { return element < 0; } ) ) {
                return std::numeric_limits<T>::max();
            }

            // Solve the ODE with the given parameters.
            ivp_pandemic_functor_obj_.set_params( p );
            ode_solver_.integrate( y_, t_, y0_, 0.125, ivp_pandemic_functor_obj_ );

            // Calculate the LSE.
            double value = 0.0;
            for ( decltype( x_obs_.size1() ) i = 1; i < x_obs_.size1(); ++i ) {
                for ( decltype( x_obs_.size2() ) j = 0; j < x_obs_.size2(); ++j ) {
                    value += ( std::pow( y_( 8*i, j ) - x_obs_( i, j ), 2 ) );
                }
            }
            return value / ( total_time_ * pop_size_squared_ );
        }

        private:
            ublas::matrix<T> x_obs_;                                        // Observation matrix.
            mutable ublas::matrix<T> y_;                                    // Solution matrix for ODE.
            mutable ublas::vector<T> t_;                                    // Time vector for ODE.
            ublas::vector<T> y0_;                                           // Initial value of observated cases.
            mutable tws::ivp_pandemic_functor<T> ivp_pandemic_functor_obj_; // ODE that the solver has to calculate.
            mutable Ode ode_solver_;                                        // ODE solver we will use.
            T pop_size_squared_;                                            // Number of people squared.
            int total_time_;                                                // Time horizon of the observation matrix.
        } ; // struct lse_functor

} // namespace tws

#endif