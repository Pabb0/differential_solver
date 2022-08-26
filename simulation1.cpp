#include <fstream>
#include <string>
#include <cassert>
#include <limits>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>

#include "ivp_solvers.hpp"
#include "ivp_pandemic.hpp"

namespace ublas = boost::numeric::ublas;

// Reads the parameters (real parameters and initial conditions)
// from the given name of a file
// into the params vector and the initial conditions vector.
template <typename T>
void read_params_from_file( std::string const & filename, ublas::vector<T> & params, ublas::vector<T> & y0 ) {
    #ifndef NDEBUG
        assert( params.size() == 5 );
        assert( y0.size() == 5 );
    #endif
    std::ifstream file( filename );
    std::string line;
    std::getline( file, line );
    std::istringstream split_line( line );

    int counter = 0;
    for( std::string s ; split_line >> s; ) {
        if ( counter < 5 ) {
            params( counter ) = stod( s );
        } else {
            y0( counter % 5 ) = stod( s );
        }
        ++counter;
    }
}

// Writes the solution with corresponding time point
// to the file which name is given in filename.
template<typename T>
void write_sol_to_file( std::string const & filename, ublas::matrix<T> const & y, ublas::vector<T> const & t ) {
    #ifndef NDEBUG
        assert( y.size1() == t.size() );
    #endif

    std::ofstream outputFile(filename);
    for ( decltype( t.size() ) i = 0; i < t.size(); ++i ) {
        outputFile << t(i) << " ";
        for ( decltype( y.size2() ) j = 0; j < y.size2(); ++j ) {
            outputFile << y( y.size2() * i + j ) << " ";
        }
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
    int system_size = 5; // Size of the system of ODEs.
    int nb_params = 5; // Number of parameters in our model.

    // Set parameters and initial conditions.
    ublas::vector<double> params( nb_params, 0.0 );
    ublas::vector<double> y0( system_size, 0.0 );
    read_params_from_file( "parameters.in", params, y0 );
    
    // Construct IVP.
    tws::ivp_pandemic_functor<double> ivp_functor_obj(params);

    // Construct necessary matrices and vectors to contain solutions.
    ublas::vector<double> t( N + 1, 0.0 );
    ublas::matrix<double> y_forward_euler( N + 1, system_size, 0.0 );
    ublas::matrix<double> y_heun( N + 1, system_size, 0.0 );
    ublas::matrix<double> y_backward_euler( N + 1, system_size, 0.0 );

    // Perform numerical computations
    // Forward Euler
    tws::forward_euler<double, decltype ( ivp_functor_obj )> fe( system_size );
    fe.integrate( y_forward_euler, t, y0, dt, ivp_functor_obj );

    // Heun
    tws::heun<double, decltype ( ivp_functor_obj )> h( system_size );
    h.integrate( y_heun, t, y0, dt, ivp_functor_obj );

    // Backward Euler
    double eps = 1000 * std::numeric_limits<double>::epsilon();
    tws::backward_euler<double, decltype ( ivp_functor_obj )> be( system_size );
    be.integrate( y_backward_euler, t, y0, dt, ivp_functor_obj, eps );

    // // Write solutions to file.
    // write_sol_to_file( "fwe_no_measures.txt", y_forward_euler, t );
    // write_sol_to_file( "heun_lockdown.txt", y_heun, t );
    write_sol_to_file( "bwe_quarantine.txt", y_backward_euler, t );

    return 0;
}