#include <string>
#include <fstream>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/assignment.hpp>
#include <cmath>

#include "lse.hpp"
#include "bfgs.hpp"
#include "ivp_pandemic.hpp"
#include "ivp_solvers.hpp"

namespace ublas = boost::numeric::ublas;

// Reads the observations in a matrix and returns it.
ublas::matrix<double> read_observation_from_file( std::string const & filename ) {
    std::ifstream file( filename );
    std::string line;
    std::getline( file, line );
    std::istringstream split_line( line );

    std::string s;
    split_line >> s;
    int N = stoi( s );
    split_line >> s;
    int system_size = stoi( s );

    ublas::matrix<double> observations( N, system_size );

    int row_counter = 0;
    int col_counter = 0;
    while ( std::getline( file, line ) ) {
         col_counter = 0;
         std::istringstream split_line( line );
         split_line >> s;
        
         for( std::string s ; split_line >> s; ){
             observations( row_counter, col_counter ) = stod( s );
             col_counter += 1;
         }
         row_counter += 1;
    } 
    return observations;
}

int main() {
    // OBSERVATIONS1.IN
    // Get observations from file.
    ublas::matrix<double> y_observations = read_observation_from_file( "observations1.in" );
    ublas::vector<double> y0 = ublas::row( y_observations, 0 );

    // Prepare initial parameters vector.
    ublas::vector<double> p( 5 );
    p( 0 ) = 0.32; p( 1 ) = 0.03; p( 2 ) = 0.151; p( 3 ) = 0.004; p( 4 ) = 0.052;

    // Prepare initial Hessian matrix (identity matrix).
    ublas::matrix<double> B( p.size(), p.size(), 0.0 );
    for ( decltype ( B.size1() ) i = 0; i < B.size1(); ++i ) B( i, i ) = 1.0;

    int nb_iters = 0;  // To store number of iterations of the bfgs algorithm.  
    double tol = 1e-7; // Tolerance of optimization

    // Define function that we want to optimize.
    tws::ivp_pandemic_functor<double> ivp_functor_obj( p );
    tws::heun<double, decltype ( ivp_functor_obj )> h( y0.size() );
    tws::lse_functor<double, decltype( h )> lse_functor_obj( y_observations, y0, p, h );

    // Perform BFGS (solution gets put in p)
    tws::bfgs( p, B, lse_functor_obj, tol, nb_iters );

    std::cout << "Obtained parameters from observations1.in are: (" << p( 0 ) << ", " << p( 1 ) << ", " << p( 2 ) << ", " << p( 3 ) << ", " << p( 4 ) << ")" << std::endl;

    // OBSERVATIONS2.IN
    // Get observations from file.
    y_observations = read_observation_from_file( "observations2.in" );
    y0 = ublas::row( y_observations, 0 );

    // Prepare initial parameters vector.
    p( 0 ) = 0.5; p( 1 ) = 0.08; p( 2 ) = 0.04; p( 3 ) = 0.004; p( 4 ) = 0.09;

    // Prepare initial Hessian matrix (identity).
    std::fill_n( B.begin2(), B.size1() * B.size2() , 0.0 );
    for ( decltype ( B.size1() ) i = 0; i < B.size1(); ++i ) B( i, i ) = 1.0;
    
    nb_iters = 0; // To store number of iterations of the bfgs algorithm.
    tol = 1e-7; // Tolerance of optimization

    // Function that we want to optimize.
    ivp_functor_obj.set_params( p );
    tws::lse_functor<double, decltype( h )> lse_functor_obj_2( y_observations, y0, p, h );

    // Perform BFGS (solution gets put in p)
    tws::bfgs( p, B, lse_functor_obj_2, tol, nb_iters );

    std::cout << "Obtained parameters from observations2.in are: (" << p( 0 ) << ", " << p( 1 ) << ", " << p( 2 ) << ", " << p( 3 ) << ", " << p( 4 ) << ")" << std::endl;
}