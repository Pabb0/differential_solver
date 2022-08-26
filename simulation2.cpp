#include <cassert>
#include <numeric>
#include <fstream>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/banded.hpp>

#include "ivp_solvers.hpp"

namespace ublas = boost::numeric::ublas;

// Possibility 1 --> Using a function.
// Function that calculates the system of ODEs.
template <typename T>
void ivp_function( ublas::vector<T> const & y, ublas::vector<T> & result ) {
    #ifndef NDEBUG
        assert( y.size() == result.size() );
    #endif

    int term = 0;
    std::transform( y.begin(), y.end(), result.begin(),
        [& term] ( auto number ) { return -10 * std::pow( number - 0.1 * ( term++ ), 3 ); } );

} // ivp_function

// Possibility 2 --> Using a functor.
// Functor that calculates the system of ODEs.
template <typename T>
struct ivp_functor {

    typedef ublas::diagonal_matrix<T> matrix_type;

    ivp_functor() {};

    void operator() ( ublas::vector<T> const & y, ublas::vector<T> & result ) const {
        #ifndef NDEBUG
            assert( y.size() == result.size() );
        #endif

        int term = 0;
        std::transform( y.begin(), y.end(), result.begin(),
            [& term] ( auto number ) { return -10 * std::pow( number - 0.1 * term++, 3 ); } );
    }

    void jacobian( ublas::vector<T> const & y, matrix_type & result ) const {
        #ifndef NDEBUG
            assert( result.size1() == result.size2() );
            assert( y.size() == result.size1() );
        #endif
        
        for ( decltype( y.size() ) i = 0; i < y.size(); ++i ) { // Iterate over the diagonal
            result( i, i ) = -3 * 10 * std::pow( y(i) - 0.1 * i, 2 );
        }
    }
} ; // struct ivp_functor


// Writes the solution with corresponding time point
// to the file which name is given in filename.
template<typename T>
void write_sol_to_file( std::string const & filename, ublas::matrix<T> const & y, ublas::vector<T> const & t, int end_time ) {
    #ifndef NDEBUG
        assert( end_time > 0);
        assert( y.size1() == t.size() );
    #endif

    std::ofstream outputFile( filename );
    int timestep_counter = ( t.size() - 1 ) / end_time;

    for ( decltype( t.size() ) i = 0; i <= end_time; ++i ) {
        outputFile << t( i * timestep_counter ) << " ";
        outputFile << y( i * timestep_counter, 0 ) << " ";
        outputFile << y( i * timestep_counter, 24) << " ";
        outputFile << y( i * timestep_counter, 49 ) << " ";
        outputFile << std::endl;
    }
}

int main( int argc, char* argv[] ) {
    #ifndef NDEBUG
        assert( argc == 3 );
    #endif

    int N = std::atoi( argv[1] ); // Number of timesteps.
    int T = std::atoi( argv[2] ); // Time horizon.

    #ifndef NDEBUG
        assert( N > 0 );
        assert( T > 0 );
    #endif

    double dt = ( (double) T ) / ( (double) N ); // Time step.
    int system_size = 50; // Size of the system of ODEs.

    // Initialise the initial condition
    ublas::vector<double> y0( system_size ); 
    std::iota( y0.begin(), y0.end(), 1 );
    y0 *= 0.01;

    // Vector and matrices that will contain the time points / solutions.
    ublas::vector<double> t( N + 1 );
    ublas::matrix<double> y_forward_euler( N + 1, system_size );
    ublas::matrix<double> y_backward_euler( N + 1, system_size );
    ublas::matrix<double> y_heun( N + 1, system_size );

    // Possibility 1 - Function (Forward Euler).
    tws::forward_euler<double, decltype( ivp_function<double> )> fe( system_size );
    fe.integrate( y_forward_euler, t, y0, dt, ivp_function<double> );
    write_sol_to_file( "fwe_sim2.txt", y_forward_euler, t, T );

    // Possibility 2 - Functor (Backward Euler).
    ivp_functor<double> ivp_functor_obj;
    tws::backward_euler<double, ivp_functor<double>> be( system_size );
    double eps = std::sqrt( std::numeric_limits<double>::epsilon() );
    be.integrate( y_backward_euler, t, y0, dt, ivp_functor_obj);
    write_sol_to_file( "bwe_sim2.txt", y_backward_euler, t, T );

    // Possibility 3 - Lambda expression (Heun).
    auto ivp_lambda = [] ( auto const &x, auto &y ) -> void { ivp_function<double>( x, y ); }; 
    tws::heun<double, decltype( ivp_lambda )> h( system_size );
    h.integrate( y_heun, t, y0, dt, ivp_lambda );
    write_sol_to_file( "heun_sim2.txt", y_heun, t, T );

    return 0;
}