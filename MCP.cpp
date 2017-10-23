#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <iomanip>
#include "mpi.h"

#define _L_ 40              //  Lattice size, defined by the number of Cells in a row, in each direction
#define _NPC_X_ 1     // Number of sub-lattices (processors) in a row along x, _L_ must be a multiple of _NPC_X_
#define _NPC_Y_ 1     // _NPC_X_ * _NPC_Y_ * _NPC_Z_ = Number of processors
#define _NPC_Z_ 1

#define _CONCENTRATION_ 0.05 // concentration of atoms of type 1 that will be put in initially 0-type cells
#define _CLUSTERS_ 0                   // 0=no cluster ; 1=clusters
//const char (_TYPE_[3])[3] = {"--", "Fe", "--"};
const double  momentum[3] = {.007, 3.763, -0.010}; // magnetic moments of each atom type (0,1,2...)

#define _N_ATOMS_PER_CELL_ 4
#define _N_SHELLS_ 24
#define _MAX_ATOMS_PER_SHELL_ 150

#define _CELL_RANGE_X_ 4      //depends on how far the interactions are considered
#define _CELL_RANGE_Y_ 4     //given in number of Cells, following one direction
#define _CELL_RANGE_Z_ 4

#define _TEMP_MIN_ 10
#define _TEMP_MAX_ 115
#define _TEMP_STEP_ 5

#define _METHOD_  5  // 0:Genuine Metropolis // 3:Heat Bath // 5:Wolff (only with 1 processor)

#define _T_MIN_ 1000 // minimal time (in MC) of both convergence phase & measurement phase
#define _T_CONVERG_ 3000
#define _T_MEASURE_ 200000
#define _GOOD_AVERAGE_ 0.0005 // the less it is, the better the averages will be

#define _USE_ONLY_MAGNETIC_ATOMS_ 1
#define _MEASURE_NRJ_ 0
#define _T_REFRESH_GHOST_AREAS_ 100

#define _Jij_SCALING_FACTOR_ 2   // You might need to scale your Jij's depending on your definition of Jij's.
//BE CAREFUL, WITH the function : int which_Jij_set()
#define _N_Jij_SETS_ 1
#define _LIMIT_ANGLE_ Pi
#define _MAX_CLUSTER_SIZE_ 12
#define _SEED_ (5+MPI_my_rank)

const double Pi = 3.1415926635898;
const double _k_ = 0.00633365936133; // Jij must here be given in mRy

//// You should not have to change the followings parameters////
int MPI_my_rank, MPI_p;
int neighbour_rank[3][3][3];
const int L_eff_x = _L_ / _NPC_X_ ; // length (along x) of a sub-lattice treated by one processor
const int L_eff_y = _L_ / _NPC_Y_ ;
const int L_eff_z = _L_ / _NPC_Z_ ;

const int L_min_x = _CELL_RANGE_X_; // length (along x) of a ghost area = begining point of the real area
const int L_min_y = _CELL_RANGE_Y_;
const int L_min_z = _CELL_RANGE_Z_;

const int L_max_x = L_min_x + L_eff_x; // end of the real area
const int L_max_y = L_min_y + L_eff_y;
const int L_max_z = L_min_z + L_eff_z;

const int L_tot_x = L_max_x + _CELL_RANGE_X_;// total length of the lattice treated by the processor (including ghost areas)
const int L_tot_y = L_max_y + _CELL_RANGE_Y_;
const int L_tot_z = L_max_z + _CELL_RANGE_Z_;

const double _cos_limit_angle_ = cos(_LIMIT_ANGLE_);

enum boolean {no, yes};

using namespace std;


////// ---   DECLARATIONS   ---- //////
double abs(double x); //might need to be commented
int random_range(int lowest_number, int highest_number);
double random_unit();
double min(double, double);
double max(double, double);
double trunc(double, int);
////// ---   ALGEBRA    ---- //////


/*Object that contains 3D integer coordinates (l,m,n)
(used for the positions the of Cells within the lattice)*/
struct coord{
 int l;
 int m;
 int n;

 coord(){}


 coord(const coord & c){
    l = c.l;
    m = c.m;
    n = c.n;
 }


 coord (int ll, int mm, int nn){
    l = ll;
    m = mm;
    n = nn;
 }

 void operator=(coord c){
    l = c.l;
    m = c.m;
    n = c.n;
 }

};


/*3D vector (x,y,z)*/
struct vect{
 double x;
 double y;
 double z;

 vect(){}

 vect(const vect & v){
    x = v.x;
    y = v.y;
    z = v.z;
 }

 vect (double xx, double yy, double zz){
    x = xx;
    y = yy;
    z = zz;
 }

 void operator=(vect v){
    x = v.x;
    y = v.y;
    z = v.z;
 }

 void operator+=(vect v){
    x += v.x;
    y += v.y;
    z += v.z;
 }

 void operator-=(vect v){
    x -= v.x;
    y -= v.y;
    z -= v.z;
 }

 void operator/=(double lambda){
   if (lambda != 0){
    x /= lambda;
    y /= lambda;
    z /= lambda;
   }
   else{ cout << "Pb of division by 0" << endl;}  
 }

 void print(){
    cout << "x =" << x <<" y =" << y << " z =" << z <<  " norm =" << sqrt(x*x+y*y+z*z)<< endl ;
}

};


struct matrix{
 double a[3][3];
};


////// --- DECLARATIONS ---- //////
coord coord_(int, int, int);
vect vect_(double, double, double);

vect operator + (vect v1, vect v2);
vect operator - (vect v1, vect v2);
coord operator - (coord c1, coord c2);
double operator * (vect v1, vect v2);
vect operator ^ (vect v1, vect v2);
vect operator + (coord c1, vect v2);
vect operator - (coord c1, vect v2);
vect operator * (vect v, double lambda);
vect operator * (double lambda,vect v);
vect operator / (vect v, double lambda);
int operator ==(vect v1, vect v2);
vect operator * (matrix mat, vect v);

boolean does_any_processor(boolean my_answer, MPI_Comm * Comm);

vect new_random_spin();
double norm_of(vect);
int which_Jij_set(vect m1, vect m2);
vect bring2zone(vect one);
vect getvect(double theta,double phi);



/////////////////////////////////////////////////////////////////////////////
//////////////////////////// ---- STATISTIC ---- ////////////////////////////
/////////////////////////////////////////////////////////////////////////////
class Statistic{

 private:
  boolean filled ;
  int n_points, n_first_points, n_average;
  double * data;

 public:
  int current;
  double average, fluctu;

 Statistic(int n){
    int i;
    n_points = n;
    n_first_points = int(n / 3);
    data = new double[n];   
    filled = no;
    average = 0;
    fluctu = 0;
    current = 0;
    for (i=0;i<n_points;i++){
     data[i] = 0;
    }
 }

 
 ~Statistic(){
    delete data;
   }
   

 void add(double new_data){  // adds a data to the table of datas
    current = (current + 1) % n_points;
    if (current==0){ filled = yes ;}
    data[current] = new_data;
 }
 
 
 // clears all datas (before a new measurement for example)
 void clear(){
    int i;
    filled = no;
    average = 0;
    fluctu = 0;
    current = 0;
    for (i=0;i<n_points;i++){
     data[i] = 0;
    }
 }
 
 
 /* evaluates the average and fluctuations, the result is stored in "average", 
 fluctuations in "fluctu", the number of points considered in "n_average"*/
 void calcul_average(){ 
    int i;
    average = 0;
    fluctu = 0;

    if (filled == yes){
     for (i=0;i<n_points;i++){
      average += data[i];
      fluctu += data[i]*data[i];
     }
     fluctu = sqrt(abs(fluctu - average * average / n_points)/n_points );
     average = average / n_points;
     n_average = n_points; // n_average represents the number of points on which the average is made
    }
    else{
     for (i=1;i<=current;i++){
      average += data[i];
      fluctu += data[i]*data[i];
     }
     fluctu = sqrt(abs(fluctu - average * average / current)/current );
     average = average / current;
     n_average = current;
    }
 }
 
 
 // evaluates how a virtual peak of 5 times the value of the average relative fluctuation, lasting over 5 datas, would change the average value
 boolean is_good_average(double threshold){
    if (((25*fluctu) / (average * (n_average + 5))) <= threshold){return yes;}else{return no;}
 }
 
 
 // evaluates the difference between local average performed over newer values and older ones
 boolean is_changing(double threshold){ 
    int i, newer, older;
    double newer_average=0, older_average=0, change;
    if (filled==yes) {
     for (i=0;i<n_first_points;i++){
      newer = (current - i + n_points) % n_points;
      older = (current + i +1) % n_points;
      newer_average += data[newer];
      older_average += data[older];
     }
     newer_average /= n_first_points;
     older_average /= n_first_points;
     change = newer_average - older_average;
     
     if (abs(change) >= threshold ){return yes;}else{return no ;}
    }else{return yes;}
 }

 
 // evaluates if the fluctuations are smaller than "threshold"
 boolean is_fluctuating(double threshold){
    if ( fluctu <= (threshold)){return no;}else{return yes;}
 }
 
 
 // prints the whole table of datas to file
 void print(ofstream *out_file){ 
    int i;
    if(out_file) {
     for (i=0;i<n_points;i++){
      *out_file << data[i] << "\n" ;
     }
     *out_file << "\n" ;
     out_file->flush();
    }
 }
};


/////////////////////////////////////////////////////////////////////////////
//////////////////////////// ----   ATOM    ---- ////////////////////////////
/////////////////////////////////////////////////////////////////////////////
class Atom{
 public:
 boolean tag;
 int type;
 vect pos;
 vect m, local_mag;
 double* Jij[_N_Jij_SETS_][_N_SHELLS_];
 Atom ***neighbour;
 void *cell;


 Atom(){
    int q;
    tag = no;
    type = 0; // ----------------------------------------------------------------ALL ATOMS ARE INITIALLY 0-type
    local_mag = vect_(0,0,0);
    neighbour = new Atom**[_N_SHELLS_]; //paranteser borttagna
    
    for (q=0 ; q<_N_SHELLS_ ; q++){
    Jij[0][q] = NULL;
    if(_N_Jij_SETS_==2){Jij[1][q] = NULL;}
    neighbour[q] = NULL;
    }    
    
 }
 
 
 ~Atom(){
    int q;
    for (q=0 ; q<_N_SHELLS_ ; q++){
    if (neighbour[q] != NULL){delete neighbour[q];}
    }
    delete neighbour;
 }

 
 // changes the spin "m" to "new_spin"
 inline void flip(vect new_spin) {
    m = new_spin;
 }


 //evaluates the local energy
 inline double nrj_neighbour(){
     get_local_mag();
     return -(local_mag * m);
 }


 // evaluates the local magnetization : \sum_j Jij \vec{e}_j
 inline void get_local_mag(){
    int q, n, Jij_set;
    local_mag = vect_(0,0,0);

    for (q=0 ; q<_N_SHELLS_ ; q++){
     n=0;
     while(neighbour[q][n] != NULL){
      Jij_set = which_Jij_set(neighbour[q][n]->m,m);
      local_mag += Jij[Jij_set][q][n] * neighbour[q][n]->m;
      n++;
     }
    }
    
 }
  
  
  //checks if the atom is in the real area or in some ghost_area
  boolean is_in_real_area(); // the definition is given afer
  
};


// pre-atom structure exchanged during the refreshing of ghost areas (MPI)
struct Atom_data {   
    double mx, my, mz;
    int type;
};


// atom structure exchanged during the refreshing of ghost areas (MPI)
void mk_mpi_Atom_data_type( MPI_Datatype *dt ) { 
    MPI_Datatype type[4] = { MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_INT};
    int blocklen[4] = { 1, 1 ,1, 1};
    MPI_Aint disp[4], base;
    struct Atom_data sample;

    MPI_Address( &sample, &base );
    MPI_Address( &sample.mx, &disp[0] );
    disp[0] -= base;
    MPI_Address( &sample.my, &disp[1] );
    disp[1] -= base;
    MPI_Address( &sample.mz, &disp[2] );
    disp[2] -= base;
    MPI_Address( &sample.type, &disp[3] );
    disp[3] -= base;

    MPI_Type_struct( 4, blocklen, disp, type, dt );
    MPI_Type_commit( dt );
}

/////////////////////////////////////////////////////////////////////////
//////////////////////////// ----   CELL    ---- ////////////////////////////
/////////////////////////////////////////////////////////////////////////
class Cell{
 public:
  Atom *atom[_N_ATOMS_PER_CELL_];
  coord pos;

  
 Cell(){
    int n;
    for(n = 0 ; n<_N_ATOMS_PER_CELL_ ; n++){atom[n] = new Atom(); atom[n]->cell = this;}
    
///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////   INITIALIZATION OF THE CELL   ////////////////////////
//Be careful, the number of atom must not exceed _N_ATOMS_PER_CELL_

     // Zn
    atom[0]->pos = vect_(0,0,0); // vectors should be given in the "basis_vector" basis.

    atom[1]->pos = vect_(0.5,0.5,0);
    atom[2]->pos = vect_(0,0.5,0.5);
    atom[3]->pos = vect_(0.5,0,0.5);

    //Te
 /*
    atom[4]->pos = vect_(0.25,0.25,0.25);
    atom[4]->type = 2;
    atom[5]->pos = vect_(0.75,0.75,0.25);
    atom[5]->type = 2;
    atom[6]->pos = vect_(0.25,0.75,0.75);
    atom[6]->type = 2;
    atom[7]->pos = vect_(0.75,0.25,0.75);
    atom[7]->type = 2;
 */
 ///////////////////////////////////////////////////////////////////////////////////////////
}


 ~Cell(){
    int n;
    for(n = 0 ; n<_N_ATOMS_PER_CELL_ ; n++){delete atom[n];}
 }

};


// pre-Cell structure exchanged during the refreshing of ghost areas (MPI)
struct Cell_data {
   Atom_data atom_data [_N_ATOMS_PER_CELL_];
};


boolean Atom::is_in_real_area(){
      Cell *my_cell = static_cast<Cell*>(cell);
      if (my_cell->pos.l < 0 ||my_cell->pos.l >= L_eff_x ||my_cell->pos.m < 0 ||my_cell->pos.m >= L_eff_y ||my_cell->pos.n < 0 ||my_cell->pos.n >= L_eff_z ){return no;}else{return yes;}
 }


 // Cell structure exchanged during the refreshing of ghost areas (MPI)
void mk_mpi_Cell_data_type(MPI_Datatype Atom_data_type,  MPI_Datatype *dt ) {
    MPI_Datatype type[1] = {Atom_data_type};
    int blocklen[1] = {_N_ATOMS_PER_CELL_};
    MPI_Aint disp[1], base;
    struct Cell_data sample;

    MPI_Address( &sample, &base );
    MPI_Address( &sample.atom_data[0], &disp[0] );
    disp[0] -= base;

    MPI_Type_struct( 1, blocklen, disp, type, dt );
    MPI_Type_commit( dt );
}


/////////////////////////////////////////////////////////////////////////////
//////////////////////////// ----  CLUSTER  ---- ////////////////////////////
/////////////////////////////////////////////////////////////////////////////
struct Atomlink{
 Atom * link;
};


class Cluster{
 private:
 public:
  int size;
  int N_overlap;
  Atomlink *atom;
  

  Cluster(){
    atom = NULL ;
 }
 

 ~Cluster(){
   if (atom != NULL) { delete atom;}
 }

 
//Checks if there are already some magnetic atoms around, if there are -> stop
 boolean is_enough_space(int s, Atom * kernel, int dest_type, int source_type){
    int n2; 
    Atom *current_atom = kernel;
    
    for (n2=0 ; n2 < s ; n2++){
     if (current_atom->type == dest_type){return no; }
     else if(current_atom->type == source_type) {}
     else {cout << "YOU ARE PUTTING MAGNETIC ATOMS IN THE WRONG PLACE ??" << endl; return no ;}
     current_atom = kernel->neighbour[0][n2];
    }
    
    return yes;
 }
 
 
 void create(int s, Atom *kernel, int dest_type, int source_type){ // "s" is the size (in number of atoms)
    int n2 ;
    Atom *current_atom = kernel;
    size = s; 
    atom = new Atomlink[s];

// SPHERICAL SHAPE (around the kernel)
    for (n2=0 ; n2 < size ; n2++){
     if(current_atom->type == source_type) {current_atom->type = dest_type;}
     atom[n2].link = current_atom;
     current_atom = kernel->neighbour[0][n2];//the magnetics clustered atoms are chosen from the first neighbour shell of the kernel (first = 0)
    }

}

};

/////////////////////////////////////////////////////////////////////////////
//////////////////////////// ---- SUB LATTICE--- ////////////////////////////
/////////////////////////////////////////////////////////////////////////////
// pre-ghost areas structure exchanged during the refreshing of ghost areas (MPI)
struct Sub_Lattice_data{
    Cell_data * cell_data;

    Sub_Lattice_data(int L,int M, int N){
      cell_data = new(Cell_data[L*M*N]);
    }
    
    ~Sub_Lattice_data(){
      delete(cell_data);
    }
};


// ghost areas structure exchanged during the refreshing of ghost areas (MPI), their size is L x M x N
void mk_mpi_Sub_Lattice_data_type(int L, int M, int N, MPI_Datatype Cell_data_type,  MPI_Datatype *dt ) {
    MPI_Datatype type[1] = {Cell_data_type};
    int blocklen[1] = {L*M*N};
    MPI_Aint disp[1], base;
    struct Sub_Lattice_data sample(L,M,N); /* used to calcuate displacements */

    MPI_Address( sample.cell_data, &base );
    MPI_Address( &(sample.cell_data[0]), &disp[0] );
    disp[0] -= base;

    MPI_Type_struct( 1, blocklen, disp, type, dt );
    MPI_Type_commit( dt );
}

/////////////////////////////////////////////////////////////////////////////
//////////////////////////// ----  LATTICE  ---- ////////////////////////////
/////////////////////////////////////////////////////////////////////////////
class Lattice{
 private:
  int length_x, length_y, length_z;
  vect basis_vector[3];
 public:
  int population, magnetic_population, population_glob, magnetic_population_glob;
  double nrj_tot, m_tot_glob, m_tot2_glob, m_tot4_glob, nrj_tot_glob, angle_deviation_tot, angle_deviation_tot_glob;
  double m_tot_vect[3], m_tot_vect_glob[3];
  vect mag;
  Cell *cell[L_tot_x][L_tot_y][L_tot_z];
  Cluster * cluster[_MAX_CLUSTER_SIZE_];
  Atomlink * magnetic_atom;
  
  // MPI declarations
  MPI_Comm MPI_Comm3D;
  void *MPI_Buffer;
  
  MPI_Datatype Atom_data_type;
  MPI_Datatype Cell_data_type;
  MPI_Datatype sub_lattice_plane_x_data_type,sub_lattice_plane_y_data_type,sub_lattice_plane_z_data_type;
  MPI_Datatype sub_lattice_line_x_data_type,sub_lattice_line_y_data_type,sub_lattice_line_z_data_type;
  MPI_Datatype sub_lattice_corner_data_type;
  
  //declarations of all the ghost areas
  Sub_Lattice_data *sub1, *sub2, *sub3, *sub4, *sub5, *sub6;
  Sub_Lattice_data *sub_line1, *sub_line2, *sub_line3, *sub_line4, *sub_line5, *sub_line6,*sub_line7, *sub_line8, *sub_line9, *sub_line10, *sub_line11, *sub_line12;
  Sub_Lattice_data *sub_corner1, *sub_corner2, *sub_corner3, *sub_corner4, *sub_corner5, *sub_corner6, *sub_corner7, *sub_corner8;
  Sub_Lattice_data *ghost_sub1, *ghost_sub2, *ghost_sub3, *ghost_sub4, *ghost_sub5, *ghost_sub6;
  Sub_Lattice_data *ghost_sub_line1, *ghost_sub_line2, *ghost_sub_line3, *ghost_sub_line4, *ghost_sub_line5, *ghost_sub_line6,*ghost_sub_line7, *ghost_sub_line8, *ghost_sub_line9, *ghost_sub_line10, *ghost_sub_line11, *ghost_sub_line12;
  Sub_Lattice_data *ghost_sub_corner1, *ghost_sub_corner2, *ghost_sub_corner3, *ghost_sub_corner4, *ghost_sub_corner5, *ghost_sub_corner6, *ghost_sub_corner7, *ghost_sub_corner8;

  MPI_Status status_p1, status_p2, status_p3, status_p4, status_p5, status_p6, status_l1, status_l2, status_l3, status_l4, status_l5, status_l6, status_l7, status_l8, status_l9, status_l10, status_l11, status_l12, status_c1, status_c2, status_c3, status_c4, status_c5, status_c6, status_c7, status_c8;


 Lattice(MPI_Comm MPI_CommX) {
    extern int MPI_my_rank;
    MPI_Comm3D = MPI_CommX;
    init();
    ////////////////////////////////////////////////////////////////////////////
    ///////////////      INITIALIZATION OF THE LATTICE     /////////////////
    //three translational basis vectors
    basis_vector[0] = vect_(1,0,0);
    basis_vector[1] = vect_(0,1,0);
    basis_vector[2] = vect_(0,0,1);
    
    create_cells();  

// put_spins_up(); // ---- to begin all spins aligned (cold start)
    put_random_spins(); // --- all shuffled (hot start)
    
    population = count_atoms_type();
    MPI_Reduce(&population,&population_glob,1, MPI_INT, MPI_SUM, 0, MPI_Comm3D);
    
    if (MPI_my_rank==0){cout << endl << "LATTICE LENGTH : " << _L_ << endl << endl << "NUMBER OF ATOMS : " << population_glob << endl ;}

    if (_CLUSTERS_ == 1) {
     refresh_ghost_areas();
     set_neighbours();
     set_clusters(1,0, _CONCENTRATION_); // set clusters of type 1 in the place of atoms of type 0
    }
    else{
     put_randomly_atoms_of_type(1,0,_CONCENTRATION_); //puts atoms of type 1 in the place of atoms of type 0
    }
  
    refresh_ghost_areas();
  
    set_neighbours(); // sets up the neighbours & Jij's
    
    make_table_of_magnetic_atoms();

    MPI_Reduce(&magnetic_population,&magnetic_population_glob,1, MPI_INT, MPI_SUM, 0, MPI_Comm3D);
    
    if (MPI_my_rank==0){cout << endl << "NUMBER OF MAGNETIC ATOMS : " << magnetic_population_glob << " (" << trunc(100 * double(magnetic_population_glob)/population_glob, 2)<<"%)"<< endl ;}
    ////////////////////////////////////////////////////////////////////////////
}

 ~Lattice() {
    clear_mem();
 }

 void init(){
    int h,i,j,size;

    length_x = L_eff_x;
    length_y = L_eff_y;
    length_z = L_eff_z;

    for(h=0;h<L_tot_x;h++){
    for(i=0;i<L_tot_y;i++){
    for(j=0;j<L_tot_z;j++){
     cell[h][i][j] = NULL;
    }}}
    
    magnetic_atom = NULL ;
    for (size = 0 ; size < _MAX_CLUSTER_SIZE_ ; size ++){
     cluster[size] = NULL;
    }
    
//construction of ghost areas
    mk_mpi_Atom_data_type(&Atom_data_type);
    mk_mpi_Cell_data_type(Atom_data_type, &Cell_data_type);
    //creating plane-of-atoms data type
    mk_mpi_Sub_Lattice_data_type(L_eff_y,L_eff_z,L_min_x,Cell_data_type, &sub_lattice_plane_x_data_type);
    mk_mpi_Sub_Lattice_data_type(L_eff_x,L_eff_z,L_min_y,Cell_data_type, &sub_lattice_plane_y_data_type);
    mk_mpi_Sub_Lattice_data_type(L_eff_x,L_eff_y,L_min_z,Cell_data_type, &sub_lattice_plane_z_data_type);
    //creating line-of-atoms data type
    mk_mpi_Sub_Lattice_data_type(L_eff_x,L_min_y,L_min_z,Cell_data_type, &sub_lattice_line_x_data_type);
    mk_mpi_Sub_Lattice_data_type(L_eff_y,L_min_z,L_min_x,Cell_data_type, &sub_lattice_line_y_data_type);
    mk_mpi_Sub_Lattice_data_type(L_eff_z,L_min_x,L_min_y,Cell_data_type, &sub_lattice_line_z_data_type);
    //creating corner atoms data type
    mk_mpi_Sub_Lattice_data_type(L_min_x,L_min_y,L_min_z,Cell_data_type, &sub_lattice_corner_data_type);
    
//6 planes
    sub1 = new Sub_Lattice_data(L_eff_x,L_eff_y,L_min_z);
    sub2 = new Sub_Lattice_data(L_eff_x,L_eff_y,L_min_z);
    sub3 = new Sub_Lattice_data(L_eff_x,L_eff_z,L_min_y);
    sub4 = new Sub_Lattice_data(L_eff_x,L_eff_z,L_min_y);
    sub5 = new Sub_Lattice_data(L_eff_y,L_eff_z,L_min_x);
    sub6 = new Sub_Lattice_data(L_eff_y,L_eff_z,L_min_x);
//6 ghost planes
    ghost_sub1 = new Sub_Lattice_data(L_eff_x,L_eff_y,L_min_z);
    ghost_sub2 = new Sub_Lattice_data(L_eff_x,L_eff_y,L_min_z);
    ghost_sub3 = new Sub_Lattice_data(L_eff_x,L_eff_z,L_min_y);
    ghost_sub4 = new Sub_Lattice_data(L_eff_x,L_eff_z,L_min_y);
    ghost_sub5 = new Sub_Lattice_data(L_eff_y,L_eff_z,L_min_x);
    ghost_sub6 = new Sub_Lattice_data(L_eff_y,L_eff_z,L_min_x);
//12 border lines
    sub_line1 = new Sub_Lattice_data(L_eff_x,L_min_y,L_min_z);
    sub_line2 = new Sub_Lattice_data(L_eff_x,L_min_y,L_min_z);
    sub_line3 = new Sub_Lattice_data(L_eff_x,L_min_y,L_min_z);
    sub_line4 = new Sub_Lattice_data(L_eff_x,L_min_y,L_min_z);
    sub_line5 = new Sub_Lattice_data(L_eff_y,L_min_x,L_min_z);
    sub_line6 = new Sub_Lattice_data(L_eff_y,L_min_x,L_min_z);
    sub_line7 = new Sub_Lattice_data(L_eff_y,L_min_x,L_min_z);
    sub_line8 = new Sub_Lattice_data(L_eff_y,L_min_x,L_min_z);
    sub_line9 = new Sub_Lattice_data(L_eff_z,L_min_x,L_min_y);
    sub_line10 = new Sub_Lattice_data(L_eff_z,L_min_x,L_min_y);
    sub_line11 = new Sub_Lattice_data(L_eff_z,L_min_x,L_min_y);
    sub_line12 = new Sub_Lattice_data(L_eff_z,L_min_x,L_min_y);
//12 ghost border lines
    ghost_sub_line1 = new Sub_Lattice_data(L_eff_x,L_min_y,L_min_z);
    ghost_sub_line2 = new Sub_Lattice_data(L_eff_x,L_min_y,L_min_z);
    ghost_sub_line3 = new Sub_Lattice_data(L_eff_x,L_min_y,L_min_z);
    ghost_sub_line4 = new Sub_Lattice_data(L_eff_x,L_min_y,L_min_z);
    ghost_sub_line5 = new Sub_Lattice_data(L_eff_y,L_min_x,L_min_z);
    ghost_sub_line6 = new Sub_Lattice_data(L_eff_y,L_min_x,L_min_z);
    ghost_sub_line7 = new Sub_Lattice_data(L_eff_y,L_min_x,L_min_z);
    ghost_sub_line8 = new Sub_Lattice_data(L_eff_y,L_min_x,L_min_z);
    ghost_sub_line9 = new Sub_Lattice_data(L_eff_z,L_min_x,L_min_y);
    ghost_sub_line10 = new Sub_Lattice_data(L_eff_z,L_min_x,L_min_y);
    ghost_sub_line11 = new Sub_Lattice_data(L_eff_z,L_min_x,L_min_y);
    ghost_sub_line12 = new Sub_Lattice_data(L_eff_z,L_min_x,L_min_y);
//8 corners
    sub_corner1 = new Sub_Lattice_data(L_eff_x,L_eff_y,L_eff_z);
    sub_corner2 = new Sub_Lattice_data(L_eff_x,L_eff_y,L_eff_z);
    sub_corner3 = new Sub_Lattice_data(L_eff_x,L_eff_y,L_eff_z);
    sub_corner4 = new Sub_Lattice_data(L_eff_x,L_eff_y,L_eff_z);
    sub_corner5 = new Sub_Lattice_data(L_eff_x,L_eff_y,L_eff_z);
    sub_corner6 = new Sub_Lattice_data(L_eff_x,L_eff_y,L_eff_z);
    sub_corner7 = new Sub_Lattice_data(L_eff_x,L_eff_y,L_eff_z);
    sub_corner8 = new Sub_Lattice_data(L_eff_x,L_eff_y,L_eff_z);
//8 ghost corners
    ghost_sub_corner1 = new Sub_Lattice_data(L_eff_x,L_eff_y,L_eff_z);
    ghost_sub_corner2 = new Sub_Lattice_data(L_eff_x,L_eff_y,L_eff_z);
    ghost_sub_corner3 = new Sub_Lattice_data(L_eff_x,L_eff_y,L_eff_z);
    ghost_sub_corner4 = new Sub_Lattice_data(L_eff_x,L_eff_y,L_eff_z);
    ghost_sub_corner5 = new Sub_Lattice_data(L_eff_x,L_eff_y,L_eff_z);
    ghost_sub_corner6 = new Sub_Lattice_data(L_eff_x,L_eff_y,L_eff_z);
    ghost_sub_corner7 = new Sub_Lattice_data(L_eff_x,L_eff_y,L_eff_z);
    ghost_sub_corner8 = new Sub_Lattice_data(L_eff_x,L_eff_y,L_eff_z);
    
// size of the buffer used for each send/receive MPI operation.
    int L_eff_max = int(max(max(L_eff_x,L_eff_y),L_eff_z));
    int _CELL_RANGE_max = int(max(max(_CELL_RANGE_X_,_CELL_RANGE_Y_),_CELL_RANGE_Z_));
    int MPI_Buffer_size = L_eff_max * L_eff_max  *_CELL_RANGE_max * _N_ATOMS_PER_CELL_ * 100 + 10000 ;
    
    MPI_Buffer = (void *)malloc(MPI_Buffer_size);
    MPI_Buffer_attach(MPI_Buffer,MPI_Buffer_size);

 }

  void create_cells(){
    int h, i, j;
    int start_x,start_y,start_z,stop_x,stop_y,stop_z;
    extern int MPI_my_rank;

    if(MPI_my_rank == 0){cout << "Creating Cells..." << endl ;}
    
    if (MPI_p ==1){ // if there is only one processor, it doesn't create ghost areas
     start_x = L_min_x ; start_y = L_min_y ; start_z = L_min_z;
     stop_x = L_max_x ; stop_y = L_max_y ; stop_z = L_max_z;
   }
   else{ // otherwise...it does
    start_x = start_y = start_z = 0;
    stop_x = L_tot_x ; stop_y = L_tot_y ; stop_z = L_tot_z;
   }

   for(h=start_x;h<stop_x;h++){
   for(i=start_y;i<stop_y;i++){
   for(j=start_z;j<stop_z;j++){
    cell[h][i][j] = new Cell();
    cell[h][i][j]->pos = coord_(h,i,j) - coord_(L_min_x, L_min_y, L_min_z);
   }}}

   if(MPI_my_rank == 0){cout << "\t \t ... Cells created." << endl ;}
 }

 // sets all spins to (1,0,0) vector
 void put_spins_up(){
    int h, i, j, n;

    for(h = L_min_x ; h < L_max_x ; h++){
    for(i = L_min_y ; i < L_max_y ; i++){
    for(j = L_min_z ; j < L_max_z ; j++){
     for(n = 0 ; n<_N_ATOMS_PER_CELL_ ; n++){
      cell[h][i][j]->atom[n]->m = vect_(1,0,0);
     }
    }}}

} 

 // shuffles spin vectors
 void put_random_spins() {
    int h, i, j, n;

    for(h=L_min_x;h<L_max_x;h++){
    for(i=L_min_y;i<L_max_y;i++){
    for(j=L_min_z;j<L_max_z;j++){
     for(n = 0 ; n<_N_ATOMS_PER_CELL_ ; n++){
      cell[h][i][j]->atom[n]->m = new_random_spin();
     }
    }}}

 }

 // transforms atoms of type "source_type" to atoms of type "dest_type" regarding of "concentration" rate
 void put_randomly_atoms_of_type(int dest_type, int source_type, double concentration){
    int h,i,j,n,number_of_atoms_left_to_put ;
    double p;
    
    number_of_atoms_left_to_put = int(double(count_atoms_type(source_type))*concentration);

    do{ 
     h = random_range(L_min_x, L_max_x) ;
     i = random_range(L_min_y, L_max_y) ;
     j = random_range(L_min_z, L_max_z) ;
     n = random_range(0, _N_ATOMS_PER_CELL_);

     if (cell[h][i][j]->atom[n]->type == source_type){
      p = random_unit();
      if (p <= _CONCENTRATION_) { cell[h][i][j]->atom[n]->type = dest_type;number_of_atoms_left_to_put--;}
      }
    }while(number_of_atoms_left_to_put > 0);
 }


 //packs useful datas in the ghost areas to be exchanged before the MPI process
 inline void pack_datas(int h_start,int h_stop, int i_start, int i_stop, int j_start, int j_stop, Sub_Lattice_data * sub){
    int h, i, j, n, where = -1;
    
    for(h = h_start ; h < h_stop ; h++){
    for(i = i_start ; i < i_stop ; i++){
    for(j = j_start ; j < j_stop ; j++){
     where ++;
     for(n = 0 ; n < _N_ATOMS_PER_CELL_; n++){
      sub->cell_data[where].atom_data[n].mx = cell[h][i][j]->atom[n]->m.x;
      sub->cell_data[where].atom_data[n].my = cell[h][i][j]->atom[n]->m.y;
      sub->cell_data[where].atom_data[n].mz = cell[h][i][j]->atom[n]->m.z;
      sub->cell_data[where].atom_data[n].type = cell[h][i][j]->atom[n]->type;
     }
    }}}

}

 //unpacks datas in the ghost areas to be exchanged after the MPI process
 inline void unpack_datas(int h_start,int h_stop, int i_start, int i_stop, int j_start, int j_stop, Sub_Lattice_data * sub){
    int h, i, j, n, where = -1;
    
    for(h = h_start ; h < h_stop ; h++){
    for(i = i_start ; i < i_stop ; i++){
    for(j = j_start ; j < j_stop ; j++){
     where ++;
     for(n = 0 ; n < _N_ATOMS_PER_CELL_; n++){
      cell[h][i][j]->atom[n]->m.x = sub->cell_data[where].atom_data[n].mx;
      cell[h][i][j]->atom[n]->m.y = sub->cell_data[where].atom_data[n].my;
      cell[h][i][j]->atom[n]->m.z = sub->cell_data[where].atom_data[n].mz;
      cell[h][i][j]->atom[n]->type = sub->cell_data[where].atom_data[n].type;
     }
    }}}

}


  inline void refresh_ghost_areas(){
     extern int neighbour_rank[3][3][3], MPI_p;
if (MPI_p == 1){return;}
    MPI_Barrier(MPI_Comm3D);
// collecting the datas of frontier atoms that are going to be sent
// border planes
    pack_datas(L_min_x, L_max_x, L_min_y, L_max_y, L_min_z ,L_min_z + L_min_z, sub1);
    pack_datas(L_min_x, L_max_x, L_min_y, L_max_y, L_max_z - L_min_z, L_max_z, sub2);
    pack_datas(L_min_x, L_max_x, L_min_y, L_min_y + L_min_y, L_min_z, L_max_z, sub3);
    pack_datas(L_min_x, L_max_x, L_max_y - L_min_y, L_max_y, L_min_z, L_max_z, sub4);
    pack_datas(L_min_x, L_min_x + L_min_x, L_min_y, L_max_y, L_min_z, L_max_z, sub5);
    pack_datas(L_max_x - L_min_x, L_max_x, L_min_y, L_max_y, L_min_z, L_max_z, sub6);
 // border lines
    pack_datas(L_min_x, L_max_x, L_min_y, L_min_y + L_min_y, L_min_z, L_min_z + L_min_z, sub_line1);
    pack_datas(L_min_x, L_max_x, L_max_y - L_min_y, L_max_y, L_min_z, L_min_z + L_min_z, sub_line2);
    pack_datas(L_min_x, L_max_x, L_min_y, L_min_y + L_min_y, L_max_z - L_min_z, L_max_z, sub_line3);
    pack_datas(L_min_x, L_max_x, L_max_y - L_min_y, L_max_y, L_max_z - L_min_z, L_max_z, sub_line4);
    pack_datas(L_min_x, L_min_x + L_min_x, L_min_y, L_max_y, L_min_z, L_min_z + L_min_z, sub_line5);
    pack_datas(L_min_x, L_min_x + L_min_x, L_min_y, L_max_y, L_max_z - L_min_z, L_max_z, sub_line6);
    pack_datas(L_max_x - L_min_x, L_max_x, L_min_y, L_max_y, L_min_z, L_min_z + L_min_z, sub_line7);
    pack_datas(L_max_x - L_min_x, L_max_x, L_min_y, L_max_y, L_max_z - L_min_z, L_max_z, sub_line8);
    pack_datas(L_min_x, L_min_x + L_min_x, L_min_y, L_min_y + L_min_y, L_min_z, L_max_z, sub_line9);
    pack_datas(L_min_x, L_min_x + L_min_x, L_max_y - L_min_y, L_max_y, L_min_z, L_max_z, sub_line10);
    pack_datas(L_max_x - L_min_x, L_max_x, L_min_y, L_min_y + L_min_y, L_min_z, L_max_z, sub_line11);
    pack_datas(L_max_x - L_min_x, L_max_x, L_max_y - L_min_y, L_max_y, L_min_z, L_max_z, sub_line12);
// corners
    pack_datas(L_min_x, L_min_x + L_min_x, L_min_y, L_min_y + L_min_y, L_min_z, L_min_z + L_min_z, sub_corner1);
    pack_datas(L_max_x - L_min_x, L_max_x ,L_min_y, L_min_y + L_min_y, L_min_z, L_min_z + L_min_z, sub_corner2);
    pack_datas(L_min_x, L_min_x + L_min_x, L_max_y - L_min_y, L_max_y, L_min_z, L_min_z + L_min_z, sub_corner3);
    pack_datas(L_max_x - L_min_x, L_max_x, L_max_y - L_min_y, L_max_y, L_min_z, L_min_z + L_min_z, sub_corner4);
    pack_datas(L_min_x, L_min_x + L_min_x, L_min_y, L_min_y + L_min_y, L_max_z - L_min_z, L_max_z, sub_corner5);
    pack_datas(L_max_x - L_min_x, L_max_x, L_min_y, L_min_y + L_min_y, L_max_z - L_min_z, L_max_z, sub_corner6);
    pack_datas(L_min_x, L_min_x + L_min_x, L_max_y - L_min_y, L_max_y, L_max_z - L_min_z, L_max_z, sub_corner7);
    pack_datas(L_max_x - L_min_x, L_max_x, L_max_y - L_min_y, L_max_y, L_max_z - L_min_z, L_max_z, sub_corner8);

// Proceeding the exchange operations
// exchanging planes
    MPI_Bsend(sub1->cell_data,1,sub_lattice_plane_z_data_type,neighbour_rank[1][1][0],101,MPI_Comm3D);
    MPI_Recv(ghost_sub2->cell_data,1,sub_lattice_plane_z_data_type,neighbour_rank[1][1][2],101, MPI_Comm3D, &status_p2);
    MPI_Bsend(sub2->cell_data,1,sub_lattice_plane_z_data_type,neighbour_rank[1][1][2],102,MPI_Comm3D);
    MPI_Recv(ghost_sub1->cell_data,1,sub_lattice_plane_z_data_type,neighbour_rank[1][1][0],102, MPI_Comm3D, &status_p1);
    MPI_Bsend(sub3->cell_data,1,sub_lattice_plane_y_data_type,neighbour_rank[1][0][1],103,MPI_Comm3D);
    MPI_Recv(ghost_sub4->cell_data,1,sub_lattice_plane_y_data_type,neighbour_rank[1][2][1],103, MPI_Comm3D, &status_p4);
    MPI_Bsend(sub4->cell_data,1,sub_lattice_plane_y_data_type,neighbour_rank[1][2][1],104,MPI_Comm3D);
    MPI_Recv(ghost_sub3->cell_data,1,sub_lattice_plane_y_data_type,neighbour_rank[1][0][1],104, MPI_Comm3D, &status_p3);
    MPI_Bsend(sub5->cell_data,1,sub_lattice_plane_x_data_type,neighbour_rank[0][1][1],105,MPI_Comm3D);
    MPI_Recv(ghost_sub6->cell_data,1,sub_lattice_plane_x_data_type,neighbour_rank[2][1][1],105, MPI_Comm3D, &status_p6);
    MPI_Bsend(sub6->cell_data,1,sub_lattice_plane_x_data_type,neighbour_rank[2][1][1],106,MPI_Comm3D);
    MPI_Recv(ghost_sub5->cell_data,1,sub_lattice_plane_x_data_type,neighbour_rank[0][1][1],106, MPI_Comm3D, &status_p5);
    MPI_Barrier(MPI_Comm3D);
// exchanging lines
    MPI_Bsend(sub_line1->cell_data,1,sub_lattice_line_x_data_type,neighbour_rank[1][0][0],201,MPI_Comm3D);
    MPI_Recv(ghost_sub_line4->cell_data,1,sub_lattice_line_x_data_type,neighbour_rank[1][2][2],201, MPI_Comm3D, &status_l4);
    MPI_Bsend(sub_line2->cell_data,1,sub_lattice_line_x_data_type,neighbour_rank[1][2][0],202,MPI_Comm3D);
    MPI_Recv(ghost_sub_line3->cell_data,1,sub_lattice_line_x_data_type,neighbour_rank[1][0][2],202, MPI_Comm3D, &status_l3);
    MPI_Bsend(sub_line3->cell_data,1,sub_lattice_line_x_data_type,neighbour_rank[1][0][2],203,MPI_Comm3D);
    MPI_Recv(ghost_sub_line2->cell_data,1,sub_lattice_line_x_data_type,neighbour_rank[1][2][0],203, MPI_Comm3D, &status_l2);
    MPI_Bsend(sub_line4->cell_data,1,sub_lattice_line_x_data_type,neighbour_rank[1][2][2],204,MPI_Comm3D);
    MPI_Recv(ghost_sub_line1->cell_data,1,sub_lattice_line_x_data_type,neighbour_rank[1][0][0],204, MPI_Comm3D, &status_l1);
    MPI_Bsend(sub_line5->cell_data,1,sub_lattice_line_y_data_type,neighbour_rank[0][1][0],205,MPI_Comm3D);
    MPI_Recv(ghost_sub_line8->cell_data,1,sub_lattice_line_y_data_type,neighbour_rank[2][1][2],205, MPI_Comm3D, &status_l8);
    MPI_Bsend(sub_line6->cell_data,1,sub_lattice_line_y_data_type,neighbour_rank[0][1][2],206,MPI_Comm3D);
    MPI_Recv(ghost_sub_line7->cell_data,1,sub_lattice_line_y_data_type,neighbour_rank[2][1][0],206, MPI_Comm3D, &status_l7);
    MPI_Bsend(sub_line7->cell_data,1,sub_lattice_line_y_data_type,neighbour_rank[2][1][0],207,MPI_Comm3D);
    MPI_Recv(ghost_sub_line6->cell_data,1,sub_lattice_line_y_data_type,neighbour_rank[0][1][2],207, MPI_Comm3D, &status_l6);
    MPI_Bsend(sub_line8->cell_data,1,sub_lattice_line_y_data_type,neighbour_rank[2][1][2],208,MPI_Comm3D);
    MPI_Recv(ghost_sub_line5->cell_data,1,sub_lattice_line_y_data_type,neighbour_rank[0][1][0],208, MPI_Comm3D, &status_l5);
    MPI_Bsend(sub_line9->cell_data,1,sub_lattice_line_z_data_type,neighbour_rank[0][0][1],209,MPI_Comm3D);
    MPI_Recv(ghost_sub_line12->cell_data,1,sub_lattice_line_z_data_type,neighbour_rank[2][2][1],209, MPI_Comm3D, &status_l12);
    MPI_Bsend(sub_line10->cell_data,1,sub_lattice_line_z_data_type,neighbour_rank[0][2][1],210,MPI_Comm3D);
    MPI_Recv(ghost_sub_line11->cell_data,1,sub_lattice_line_z_data_type,neighbour_rank[2][0][1],210, MPI_Comm3D, &status_l11);
    MPI_Bsend(sub_line11->cell_data,1,sub_lattice_line_z_data_type,neighbour_rank[2][0][1],211,MPI_Comm3D);
    MPI_Recv(ghost_sub_line10->cell_data,1,sub_lattice_line_z_data_type,neighbour_rank[0][2][1],211, MPI_Comm3D, &status_l10);
    MPI_Bsend(sub_line12->cell_data,1,sub_lattice_line_z_data_type,neighbour_rank[2][2][1],212,MPI_Comm3D);
    MPI_Recv(ghost_sub_line9->cell_data,1,sub_lattice_line_z_data_type,neighbour_rank[0][0][1],212, MPI_Comm3D, &status_l9);
    MPI_Barrier(MPI_Comm3D);
// exchanging corners
    MPI_Bsend(sub_corner1->cell_data,1,sub_lattice_corner_data_type,neighbour_rank[0][0][0],301,MPI_Comm3D);
    MPI_Recv(ghost_sub_corner8->cell_data,1,sub_lattice_corner_data_type,neighbour_rank[2][2][2],301, MPI_Comm3D, &status_c8);
    MPI_Bsend(sub_corner2->cell_data,1,sub_lattice_corner_data_type,neighbour_rank[2][0][0],302,MPI_Comm3D);
    MPI_Recv(ghost_sub_corner7->cell_data,1,sub_lattice_corner_data_type,neighbour_rank[0][2][2],302, MPI_Comm3D, &status_c7);
    MPI_Bsend(sub_corner3->cell_data,1,sub_lattice_corner_data_type,neighbour_rank[0][2][0],303,MPI_Comm3D);
    MPI_Recv(ghost_sub_corner6->cell_data,1,sub_lattice_corner_data_type,neighbour_rank[2][0][2],303, MPI_Comm3D, &status_c6);
    MPI_Bsend(sub_corner4->cell_data,1,sub_lattice_corner_data_type,neighbour_rank[2][2][0],304,MPI_Comm3D);
    MPI_Recv(ghost_sub_corner5->cell_data,1,sub_lattice_corner_data_type,neighbour_rank[0][0][2],304, MPI_Comm3D, &status_c5);
    MPI_Bsend(sub_corner5->cell_data,1,sub_lattice_corner_data_type,neighbour_rank[0][0][2],305,MPI_Comm3D);
    MPI_Recv(ghost_sub_corner4->cell_data,1,sub_lattice_corner_data_type,neighbour_rank[2][2][0],305, MPI_Comm3D, &status_c4);
    MPI_Bsend(sub_corner6->cell_data,1,sub_lattice_corner_data_type,neighbour_rank[2][0][2],306,MPI_Comm3D);
    MPI_Recv(ghost_sub_corner3->cell_data,1,sub_lattice_corner_data_type,neighbour_rank[0][2][0],306, MPI_Comm3D, &status_c3);
    MPI_Bsend(sub_corner7->cell_data,1,sub_lattice_corner_data_type,neighbour_rank[0][2][2],307,MPI_Comm3D);
    MPI_Recv(ghost_sub_corner2->cell_data,1,sub_lattice_corner_data_type,neighbour_rank[2][0][0],307, MPI_Comm3D, &status_c2);
    MPI_Bsend(sub_corner8->cell_data,1,sub_lattice_corner_data_type,neighbour_rank[2][2][2],308,MPI_Comm3D);
    MPI_Recv(ghost_sub_corner1->cell_data,1,sub_lattice_corner_data_type,neighbour_rank[0][0][0],308, MPI_Comm3D, &status_c1);

// updating the new ghost areas that where received
// border planes
    unpack_datas(L_min_x, L_max_x, L_min_y, L_max_y, 0, L_min_z, ghost_sub1);
    unpack_datas(L_min_x, L_max_x, L_min_y, L_max_y, L_max_z, L_tot_z, ghost_sub2);
    unpack_datas(L_min_x, L_max_x, 0, L_min_y, L_min_z, L_max_z, ghost_sub3);
    unpack_datas(L_min_x, L_max_x, L_max_y, L_tot_y, L_min_z, L_max_z, ghost_sub4);
    unpack_datas(0, L_min_x, L_min_y, L_max_y, L_min_z, L_max_z, ghost_sub5);
    unpack_datas(L_max_x, L_tot_x, L_min_y, L_max_y, L_min_z, L_max_z, ghost_sub6);
// border lines
    unpack_datas(L_min_x, L_max_x, 0, L_min_y , 0, L_min_z, ghost_sub_line1);
    unpack_datas(L_min_x, L_max_x, L_max_y, L_tot_y, 0, L_min_z, ghost_sub_line2);
    unpack_datas(L_min_x, L_max_x, 0, L_min_y, L_max_z, L_tot_z, ghost_sub_line3);
    unpack_datas(L_min_x, L_max_x, L_max_y, L_tot_y, L_max_z, L_tot_z, ghost_sub_line4);
    unpack_datas(0, L_min_x, L_min_y, L_max_y, 0, L_min_z, ghost_sub_line5);
    unpack_datas(0, L_min_x, L_min_y, L_max_y, L_max_z, L_tot_z, ghost_sub_line6);
    unpack_datas(L_max_x, L_tot_x, L_min_y, L_max_y, 0, L_min_z, ghost_sub_line7);
    unpack_datas(L_max_x, L_tot_x, L_min_y, L_max_y, L_max_z, L_tot_z, ghost_sub_line8);
    unpack_datas(0, L_min_x, 0, L_min_y, L_min_z, L_max_z, ghost_sub_line9);
    unpack_datas(0, L_min_x, L_max_y, L_tot_y, L_min_z, L_max_z, ghost_sub_line10);
    unpack_datas(L_max_x, L_tot_x, 0, L_min_y, L_min_z, L_max_z, ghost_sub_line11);
    unpack_datas(L_max_x, L_tot_x, L_max_y, L_tot_y, L_min_z, L_max_z, ghost_sub_line12);
// corners
    unpack_datas(0, L_min_x , 0, L_min_y, 0, L_min_z, sub_corner1);
    unpack_datas(L_max_x, L_tot_x, 0, L_min_y, 0, L_min_z, sub_corner2);
    unpack_datas(0, L_min_x, L_max_y, L_tot_y, 0, L_min_z, sub_corner3);
    unpack_datas(L_max_x, L_tot_x, L_max_y, L_tot_y, 0, L_min_z, sub_corner4);
    unpack_datas(0, L_min_x, 0, L_min_y, L_max_z, L_tot_z, sub_corner5);
    unpack_datas(L_max_x, L_tot_x, 0, L_min_y, L_max_z, L_tot_z, sub_corner6);
    unpack_datas(0, L_min_x, L_max_y, L_tot_y, L_max_z, L_tot_z, sub_corner7);
    unpack_datas(L_max_x, L_tot_x, L_max_y, L_tot_y, L_max_z, L_tot_z, sub_corner8);
 
    MPI_Barrier(MPI_Comm3D);
 }


 //unpacks type-datas (only "type") from ghost areas to real areas, used during the cluster setup
 inline void unpack_type(int h_start,int h_stop, int i_start, int i_stop, int j_start, int j_stop, Sub_Lattice_data * sub, int type){
    int h, i, j, n, where = -1;
    for(h = h_start ; h < h_stop ; h++){
    for(i = i_start ; i < i_stop ; i++){
    for(j = j_start ; j < j_stop ; j++){
     where ++;
     for(n = 0 ; n < _N_ATOMS_PER_CELL_; n++){
      if (sub->cell_data[where].atom_data[n].type==type){
       cell[h][i][j]->atom[n]->type = type;
       }
      }
    }}}
 }

 /*copies atoms type of type "type" from the ghost areas to their original source.... 
 this is (only?) used during the setting of clusters where ghost areas can be written on*/
 void ghost_areas_2_real(int type){
 if (MPI_p == 1){return;}
    extern int neighbour_rank[3][3][3];  
    MPI_Barrier(MPI_Comm3D);
// collecting the datas of frontier atoms that are going to be sent
// border planes
    pack_datas(L_min_x, L_max_x, L_min_y, L_max_y, 0, L_min_z, ghost_sub1);
    pack_datas(L_min_x, L_max_x, L_min_y, L_max_y, L_max_z, L_tot_z, ghost_sub2);
    pack_datas(L_min_x, L_max_x, 0, L_min_y, L_min_z, L_max_z, ghost_sub3);
    pack_datas(L_min_x, L_max_x, L_max_y, L_tot_y, L_min_z, L_max_z, ghost_sub4);
    pack_datas(0, L_min_x, L_min_y, L_max_y, L_min_z, L_max_z, ghost_sub5);
    pack_datas(L_max_x, L_tot_x, L_min_y, L_max_y, L_min_z, L_max_z, ghost_sub6);
// border lines
    pack_datas(L_min_x, L_max_x, 0, L_min_y , 0, L_min_z, ghost_sub_line1);
    pack_datas(L_min_x, L_max_x, L_max_y, L_tot_y, 0, L_min_z, ghost_sub_line2);
    pack_datas(L_min_x, L_max_x, 0, L_min_y, L_max_z, L_tot_z, ghost_sub_line3);
    pack_datas(L_min_x, L_max_x, L_max_y, L_tot_y, L_max_z, L_tot_z, ghost_sub_line4);
    pack_datas(0, L_min_x, L_min_y, L_max_y, 0, L_min_z, ghost_sub_line5);
    pack_datas(0, L_min_x, L_min_y, L_max_y, L_max_z, L_tot_z, ghost_sub_line6);
    pack_datas(L_max_x, L_tot_x, L_min_y, L_max_y, 0, L_min_z, ghost_sub_line7);
    pack_datas(L_max_x, L_tot_x, L_min_y, L_max_y, L_max_z, L_tot_z, ghost_sub_line8);
    pack_datas(0, L_min_x, 0, L_min_y, L_min_z, L_max_z, ghost_sub_line9);
    pack_datas(0, L_min_x, L_max_y, L_tot_y, L_min_z, L_max_z, ghost_sub_line10);
    pack_datas(L_max_x, L_tot_x, 0, L_min_y, L_min_z, L_max_z, ghost_sub_line11);
    pack_datas(L_max_x, L_tot_x, L_max_y, L_tot_y, L_min_z, L_max_z, ghost_sub_line12);
// corners
    pack_datas(0, L_min_x , 0, L_min_y, 0, L_min_z, sub_corner1);
    pack_datas(L_max_x, L_tot_x, 0, L_min_y, 0, L_min_z, sub_corner2);
    pack_datas(0, L_min_x, L_max_y, L_tot_y, 0, L_min_z, sub_corner3);
    pack_datas(L_max_x, L_tot_x, L_max_y, L_tot_y, 0, L_min_z, sub_corner4);
    pack_datas(0, L_min_x, 0, L_min_y, L_max_z, L_tot_z, sub_corner5);
    pack_datas(L_max_x, L_tot_x, 0, L_min_y, L_max_z, L_tot_z, sub_corner6);
    pack_datas(0, L_min_x, L_max_y, L_tot_y, L_max_z, L_tot_z, sub_corner7);
    pack_datas(L_max_x, L_tot_x, L_max_y, L_tot_y, L_max_z, L_tot_z, sub_corner8);

// Proceeding the exchange operations
// exchanging planes
    MPI_Bsend(ghost_sub1->cell_data,1,sub_lattice_plane_z_data_type,neighbour_rank[1][1][0],101,MPI_Comm3D);
    MPI_Recv(sub2->cell_data,1,sub_lattice_plane_z_data_type,neighbour_rank[1][1][2],101, MPI_Comm3D, &status_p2);
    MPI_Bsend(ghost_sub2->cell_data,1,sub_lattice_plane_z_data_type,neighbour_rank[1][1][2],102,MPI_Comm3D);
    MPI_Recv(sub1->cell_data,1,sub_lattice_plane_z_data_type,neighbour_rank[1][1][0],102, MPI_Comm3D, &status_p1);
    MPI_Bsend(ghost_sub3->cell_data,1,sub_lattice_plane_y_data_type,neighbour_rank[1][0][1],103,MPI_Comm3D);
    MPI_Recv(sub4->cell_data,1,sub_lattice_plane_y_data_type,neighbour_rank[1][2][1],103, MPI_Comm3D, &status_p4);
    MPI_Bsend(ghost_sub4->cell_data,1,sub_lattice_plane_y_data_type,neighbour_rank[1][2][1],104,MPI_Comm3D);
    MPI_Recv(sub3->cell_data,1,sub_lattice_plane_y_data_type,neighbour_rank[1][0][1],104, MPI_Comm3D, &status_p3);
    MPI_Bsend(ghost_sub5->cell_data,1,sub_lattice_plane_x_data_type,neighbour_rank[0][1][1],105,MPI_Comm3D);
    MPI_Recv(sub6->cell_data,1,sub_lattice_plane_x_data_type,neighbour_rank[2][1][1],105, MPI_Comm3D, &status_p6);
    MPI_Bsend(ghost_sub6->cell_data,1,sub_lattice_plane_x_data_type,neighbour_rank[2][1][1],106,MPI_Comm3D);
    MPI_Recv(sub5->cell_data,1,sub_lattice_plane_x_data_type,neighbour_rank[0][1][1],106, MPI_Comm3D, &status_p5);
    MPI_Barrier(MPI_Comm3D);
// exchanging lines
    MPI_Bsend(ghost_sub_line1->cell_data,1,sub_lattice_line_x_data_type,neighbour_rank[1][0][0],201,MPI_Comm3D);
    MPI_Recv(sub_line4->cell_data,1,sub_lattice_line_x_data_type,neighbour_rank[1][2][2],201, MPI_Comm3D, &status_l4);
    MPI_Bsend(ghost_sub_line2->cell_data,1,sub_lattice_line_x_data_type,neighbour_rank[1][2][0],202,MPI_Comm3D);
    MPI_Recv(sub_line3->cell_data,1,sub_lattice_line_x_data_type,neighbour_rank[1][0][2],202, MPI_Comm3D, &status_l3);
    MPI_Bsend(ghost_sub_line3->cell_data,1,sub_lattice_line_x_data_type,neighbour_rank[1][0][2],203,MPI_Comm3D);
    MPI_Recv(sub_line2->cell_data,1,sub_lattice_line_x_data_type,neighbour_rank[1][2][0],203, MPI_Comm3D, &status_l2);
    MPI_Bsend(ghost_sub_line4->cell_data,1,sub_lattice_line_x_data_type,neighbour_rank[1][2][2],204,MPI_Comm3D);
    MPI_Recv(sub_line1->cell_data,1,sub_lattice_line_x_data_type,neighbour_rank[1][0][0],204, MPI_Comm3D, &status_l1);
    MPI_Bsend(ghost_sub_line5->cell_data,1,sub_lattice_line_y_data_type,neighbour_rank[0][1][0],205,MPI_Comm3D);
    MPI_Recv(sub_line8->cell_data,1,sub_lattice_line_y_data_type,neighbour_rank[2][1][2],205, MPI_Comm3D, &status_l8);
    MPI_Bsend(ghost_sub_line6->cell_data,1,sub_lattice_line_y_data_type,neighbour_rank[0][1][2],206,MPI_Comm3D);
    MPI_Recv(sub_line7->cell_data,1,sub_lattice_line_y_data_type,neighbour_rank[2][1][0],206, MPI_Comm3D, &status_l7);
    MPI_Bsend(ghost_sub_line7->cell_data,1,sub_lattice_line_y_data_type,neighbour_rank[2][1][0],207,MPI_Comm3D);
    MPI_Recv(sub_line6->cell_data,1,sub_lattice_line_y_data_type,neighbour_rank[0][1][2],207, MPI_Comm3D, &status_l6);
    MPI_Bsend(ghost_sub_line8->cell_data,1,sub_lattice_line_y_data_type,neighbour_rank[2][1][2],208,MPI_Comm3D);
    MPI_Recv(sub_line5->cell_data,1,sub_lattice_line_y_data_type,neighbour_rank[0][1][0],208, MPI_Comm3D, &status_l5);
    MPI_Bsend(ghost_sub_line9->cell_data,1,sub_lattice_line_z_data_type,neighbour_rank[0][0][1],209,MPI_Comm3D);
    MPI_Recv(sub_line12->cell_data,1,sub_lattice_line_z_data_type,neighbour_rank[2][2][1],209, MPI_Comm3D, &status_l12);
    MPI_Bsend(ghost_sub_line10->cell_data,1,sub_lattice_line_z_data_type,neighbour_rank[0][2][1],210,MPI_Comm3D);
    MPI_Recv(sub_line11->cell_data,1,sub_lattice_line_z_data_type,neighbour_rank[2][0][1],210, MPI_Comm3D, &status_l11);
    MPI_Bsend(ghost_sub_line11->cell_data,1,sub_lattice_line_z_data_type,neighbour_rank[2][0][1],211,MPI_Comm3D);
    MPI_Recv(sub_line10->cell_data,1,sub_lattice_line_z_data_type,neighbour_rank[0][2][1],211, MPI_Comm3D, &status_l10);
    MPI_Bsend(ghost_sub_line12->cell_data,1,sub_lattice_line_z_data_type,neighbour_rank[2][2][1],212,MPI_Comm3D);
    MPI_Recv(sub_line9->cell_data,1,sub_lattice_line_z_data_type,neighbour_rank[0][0][1],212, MPI_Comm3D, &status_l9);
    MPI_Barrier(MPI_Comm3D);
// exchanging corners
    MPI_Bsend(ghost_sub_corner1->cell_data,1,sub_lattice_corner_data_type,neighbour_rank[0][0][0],301,MPI_Comm3D);
    MPI_Recv(sub_corner8->cell_data,1,sub_lattice_corner_data_type,neighbour_rank[2][2][2],301, MPI_Comm3D, &status_c8);
    MPI_Bsend(ghost_sub_corner2->cell_data,1,sub_lattice_corner_data_type,neighbour_rank[2][0][0],302,MPI_Comm3D);
    MPI_Recv(sub_corner7->cell_data,1,sub_lattice_corner_data_type,neighbour_rank[0][2][2],302, MPI_Comm3D, &status_c7);
    MPI_Bsend(ghost_sub_corner3->cell_data,1,sub_lattice_corner_data_type,neighbour_rank[0][2][0],303,MPI_Comm3D);
    MPI_Recv(sub_corner6->cell_data,1,sub_lattice_corner_data_type,neighbour_rank[2][0][2],303, MPI_Comm3D, &status_c6);
    MPI_Bsend(ghost_sub_corner4->cell_data,1,sub_lattice_corner_data_type,neighbour_rank[2][2][0],304,MPI_Comm3D);
    MPI_Recv(sub_corner5->cell_data,1,sub_lattice_corner_data_type,neighbour_rank[0][0][2],304, MPI_Comm3D, &status_c5);
    MPI_Bsend(ghost_sub_corner5->cell_data,1,sub_lattice_corner_data_type,neighbour_rank[0][0][2],305,MPI_Comm3D);
    MPI_Recv(sub_corner4->cell_data,1,sub_lattice_corner_data_type,neighbour_rank[2][2][0],305, MPI_Comm3D, &status_c4);
    MPI_Bsend(ghost_sub_corner6->cell_data,1,sub_lattice_corner_data_type,neighbour_rank[2][0][2],306,MPI_Comm3D);
    MPI_Recv(sub_corner3->cell_data,1,sub_lattice_corner_data_type,neighbour_rank[0][2][0],306, MPI_Comm3D, &status_c3);
    MPI_Bsend(ghost_sub_corner7->cell_data,1,sub_lattice_corner_data_type,neighbour_rank[0][2][2],307,MPI_Comm3D);
    MPI_Recv(sub_corner2->cell_data,1,sub_lattice_corner_data_type,neighbour_rank[2][0][0],307, MPI_Comm3D, &status_c2);
    MPI_Bsend(ghost_sub_corner8->cell_data,1,sub_lattice_corner_data_type,neighbour_rank[2][2][2],308,MPI_Comm3D);
    MPI_Recv(sub_corner1->cell_data,1,sub_lattice_corner_data_type,neighbour_rank[0][0][0],308, MPI_Comm3D, &status_c1);

// updating the real areas without deleting any former magnetic atom
 // border planes
    unpack_type(L_min_x, L_max_x, L_min_y, L_max_y, L_min_z ,L_min_z + L_min_z, sub1, type);
    unpack_type(L_min_x, L_max_x, L_min_y, L_max_y, L_max_z - L_min_z, L_max_z, sub2, type);
    unpack_type(L_min_x, L_max_x, L_min_y, L_min_y + L_min_y, L_min_z, L_max_z, sub3, type);
    unpack_type(L_min_x, L_max_x, L_max_y - L_min_y, L_max_y, L_min_z, L_max_z, sub4, type);
    unpack_type(L_min_x, L_min_x + L_min_x, L_min_y, L_max_y, L_min_z, L_max_z, sub5, type);
    unpack_type(L_max_x - L_min_x, L_max_x, L_min_y, L_max_y, L_min_z, L_max_z, sub6, type);
// border lines
    unpack_type(L_min_x, L_max_x, L_min_y, L_min_y + L_min_y, L_min_z, L_min_z + L_min_z, sub_line1, type);
    unpack_type(L_min_x, L_max_x, L_max_y - L_min_y, L_max_y, L_min_z, L_min_z + L_min_z, sub_line2, type);
    unpack_type(L_min_x, L_max_x, L_min_y, L_min_y + L_min_y, L_max_z - L_min_z, L_max_z, sub_line3, type);
    unpack_type(L_min_x, L_max_x, L_max_y - L_min_y, L_max_y, L_max_z - L_min_z, L_max_z, sub_line4, type);
    unpack_type(L_min_x, L_min_x + L_min_x, L_min_y, L_max_y, L_min_z, L_min_z + L_min_z, sub_line5, type);
    unpack_type(L_min_x, L_min_x + L_min_x, L_min_y, L_max_y, L_max_z - L_min_z, L_max_z, sub_line6, type);
    unpack_type(L_max_x - L_min_x, L_max_x, L_min_y, L_max_y, L_min_z, L_min_z + L_min_z, sub_line7, type);
    unpack_type(L_max_x - L_min_x, L_max_x, L_min_y, L_max_y, L_max_z - L_min_z, L_max_z, sub_line8, type);
    unpack_type(L_min_x, L_min_x + L_min_x, L_min_y, L_min_y + L_min_y, L_min_z, L_max_z, sub_line9, type);
    unpack_type(L_min_x, L_min_x + L_min_x, L_max_y - L_min_y, L_max_y, L_min_z, L_max_z, sub_line10, type);
    unpack_type(L_max_x - L_min_x, L_max_x, L_min_y, L_min_y + L_min_y, L_min_z, L_max_z, sub_line11, type);
    unpack_type(L_max_x - L_min_x, L_max_x, L_max_y - L_min_y, L_max_y, L_min_z, L_max_z, sub_line12, type);
// corners
    unpack_type(L_min_x, L_min_x + L_min_x, L_min_y, L_min_y + L_min_y, L_min_z, L_min_z + L_min_z, sub_corner1, type);
    unpack_type(L_max_x - L_min_x, L_max_x ,L_min_y, L_min_y + L_min_y, L_min_z, L_min_z + L_min_z, sub_corner2, type);
    unpack_type(L_min_x, L_min_x + L_min_x, L_max_y - L_min_y, L_max_y, L_min_z, L_min_z + L_min_z, sub_corner3, type);
    unpack_type(L_max_x - L_min_x, L_max_x, L_max_y - L_min_y, L_max_y, L_min_z, L_min_z + L_min_z, sub_corner4, type);
    unpack_type(L_min_x, L_min_x + L_min_x, L_min_y, L_min_y + L_min_y, L_max_z - L_min_z, L_max_z, sub_corner5, type);
    unpack_type(L_max_x - L_min_x, L_max_x, L_min_y, L_min_y + L_min_y, L_max_z - L_min_z, L_max_z, sub_corner6, type);
    unpack_type(L_min_x, L_min_x + L_min_x, L_max_y - L_min_y, L_max_y, L_max_z - L_min_z, L_max_z, sub_corner7, type);
    unpack_type(L_max_x - L_min_x, L_max_x, L_max_y - L_min_y, L_max_y, L_max_z - L_min_z, L_max_z, sub_corner8, type);
    
    MPI_Barrier(MPI_Comm3D);
 }



 void set_clusters(int dest_type, int source_type, double concentration){
     boolean space, need_2_refresh, still_work;
     int h, i, j, n, size, n_s[_MAX_CLUSTER_SIZE_], n_s_copy[_MAX_CLUSTER_SIZE_], N_left, N_clusters, N_clusters_glob,n_check=0, n3,count_tries;
     extern int MPI_my_rank;
     double buffer=0, distance2border;

     if(MPI_my_rank == 0){cout << "Creating clusters ..." << endl ;}
     
     N_left = int(concentration * count_atoms_type(source_type));

     // Here n_s[s] represents the percentage of clusters of size s+1
     ////////////////////   CLUSTER SIZE DISTRIBUTION    ///////////////////////////////////
     ///////////////////////////////////////////////////////////////////////////////////////////
     n_s[0] = 40;  		/////////////
     n_s[1] = 20;   		/////////////
     n_s[2] = 10;   		/////////////
     n_s[3] = 8;   		/////////////
     n_s[4] = 7;   		/////////////
     n_s[5] = 5;   		/////////////
     n_s[6] = 4;   		/////////////
     n_s[7] = 3;   		/////////////              In
     n_s[8] = 2;   		/////////////         percentage
     n_s[9] = 1;   		///////////// (total must be equal to 100)
     n_s[10] = 0;   		/////////////
     n_s[11] = 0;   		/////////////
     ///////////////////////////////////////////////////////////////////////////////////////////
 
     for (size = 0 ; size < _MAX_CLUSTER_SIZE_ ; size++) {buffer += double((size+1) * n_s[size]) /100 ;}
     N_clusters = int(N_left/buffer);  // estimates the number of clusters

     for (size = 0 ; size < _MAX_CLUSTER_SIZE_ ; size++){
      n_s[size] = int(N_clusters * double(n_s[size])/100); // From now on, n_s[s] represents the number of clusters of size s
      n_check += n_s[size] * (size+1);   // recounts the number of clustered atoms
     }
     n_s[0] += N_left - n_check ;  // puts the last atoms left in 1-size clusters, this changes a little bit the distribution...

     for (size =0 ; size < _MAX_CLUSTER_SIZE_ ; size++){ // copies n_s into n_s_copy
      n_s_copy[size] = n_s[size];
     }


     for (size = _MAX_CLUSTER_SIZE_ - 1  ; size >= 0 ; size--){
      MPI_Barrier(MPI_Comm3D);

      if (n_s[size] > 0) {cluster[size] = new Cluster[n_s[size]];} else {cluster[size] = NULL;}
      n3=0;
      count_tries = 0;

      do{
       if(n_s_copy[size] > 0){
        do{
         count_tries ++;
         if (count_tries > 2 * population) {cout << "! IMPOSSIBLE TO PUT CLUSTER OF SIZE " << size +1 << " AFTER " <<2 * population<<" tries !" << endl; break;}
         h = random_range(L_min_x, L_max_x)  ; // Selection of the cluster kernel location (at random)
         i = random_range(L_min_y , L_max_y) ;
         j = random_range(L_min_z , L_max_z) ;
         n = random_range(0, _N_ATOMS_PER_CELL_); 
         if (cell[h][i][j]->atom[n]->type != source_type){continue;} // only starts a cluster with atom of type "source_type"
	 
         space =  (cluster[size])[n3].is_enough_space(size+1, cell[h][i][j]->atom[n], dest_type, source_type); // checks if there is space to put the new cluster

        }while(space == no);// if there were already some magnetic atoms around, retry elsewhere
        if (space==no){break;}

        (cluster[size])[n3].create(size+1, cell[h][i][j]->atom[n], dest_type, source_type); // create the cluster

        n_s_copy[size]-- ;
        n3++;

        distance2border = how_far_from_border_is(cell[h][i][j]->atom[n]);
        /* if the cluster kernel is very close to one border, some magnetic atoms (type=1) might have just been put in the ghost areas, 
        then it is needed to refresh their original source in order to take into account those magnetic atoms. Moreover, in this case the
        original source of ghost areas have changed, and though ghost areas need to be refreshed.
        (the following "<= 1" corresponds to the maximal exent of a cluster...BE CAREFUL if cluster maximal size is changed)*/
        if (distance2border <= 1){
         need_2_refresh = yes; // this sub-lattice requiers a global refreshing of real & ghost areas
        }else{need_2_refresh = no;}
       }
       
       //looks if one sub-lattice has required a refreshing...and if so, perform a global refreshing
       need_2_refresh = does_any_processor(need_2_refresh, &MPI_Comm3D);
       
       if(need_2_refresh == yes){
        ghost_areas_2_real(1);
        refresh_ghost_areas();
       } 

       if (n_s_copy[size] > 0){still_work = yes;}else{still_work = no;}
       still_work = does_any_processor(still_work, &MPI_Comm3D);
      
      }while(still_work == yes);
     }

     N_clusters = 0;
     for (size =0 ; size < _MAX_CLUSTER_SIZE_ ; size++){ N_clusters += n_s[size];} // Re-counts the final number of clusters

     MPI_Reduce(&N_clusters, &N_clusters_glob, 1, MPI_INT, MPI_SUM, 0, MPI_Comm3D);
     MPI_Reduce(&n_s, &n_s_copy, _MAX_CLUSTER_SIZE_, MPI_INT, MPI_SUM, 0, MPI_Comm3D);

     if(MPI_my_rank == 0){
     cout << "NUMBER OF CLUSTERS : " << N_clusters_glob << endl;
     for (size =0 ; size < _MAX_CLUSTER_SIZE_  ; size++){
      cout << "\t" << "Number of " << (size+1) << "-atom clusters : " << n_s_copy[size] << "\t ("<< trunc(int(10000 * double(n_s_copy[size])  / N_clusters_glob),2)/100 << "%)"<<endl ;
     }
     cout << "\t" << "\t"  << "... Clusters created." << endl ;
     }

 }

//creates magnetic_atom array
 void make_table_of_magnetic_atoms(int magnetic_type=1){
    int h, i, j, n , n2=0;
    
    magnetic_population = count_atoms_type(magnetic_type);
    if (magnetic_atom != NULL){delete magnetic_atom;}
    magnetic_atom = new Atomlink[magnetic_population];
    for(h=L_min_x;h<L_max_x;h++){
    for(i=L_min_y;i<L_max_y;i++){
    for(j=L_min_z;j<L_max_z;j++){
     for(n = 0 ; n<_N_ATOMS_PER_CELL_ ; n++){
     if (cell[h][i][j]->atom[n]->type == magnetic_type){magnetic_atom[n2++].link = cell[h][i][j]->atom[n];}
     }
    }}}
}

 int count_atoms_type(int type = -137){ // if no "type" is put, returns the total number of atoms of type >= 0
    int h, i, j, n , n2=0;
    for(h=L_min_x;h<L_max_x;h++){
    for(i=L_min_y;i<L_max_y;i++){
    for(j=L_min_z;j<L_max_z;j++){
     for(n = 0 ; n<_N_ATOMS_PER_CELL_ ; n++){
      if (type == -137){
       if (cell[h][i][j]->atom[n]->type >= 0){n2++;} // Counts all atoms
      }else{
       if (cell[h][i][j]->atom[n]->type == type){n2++;} // count alls atoms of type "type"
      }
     }
    }}}
    return n2;
 }



//Creates the table of neighbours of each atom, and attribuates the Jij's
 void set_neighbours() {
    int h, i, j, ht, it, jt, hh, jj, ii, n, n2, q, q2, num_shell, number[_N_SHELLS_], type_i, type_j;
    extern int MPI_my_rank;
    double Jij, Jij_2, Jij_buffer[_N_Jij_SETS_][_N_SHELLS_][_MAX_ATOMS_PER_SHELL_];
    vect R_ij;
    Atom *atom_i, *atom_j,* neighbour_buffer [_N_SHELLS_][_MAX_ATOMS_PER_SHELL_];
    
    if(MPI_my_rank == 0){    cout << endl <<"Setting up neighbours, and corresponding Jij's..."  <<endl ;}

    for(h=L_min_x;h<L_max_x;h++){ // Loops over each cell.
    for(i=L_min_y;i<L_max_y;i++){
    for(j=L_min_z;j<L_max_z;j++){
     for(n = 0 ; n<_N_ATOMS_PER_CELL_ ; n++){
     
      atom_i = cell[h][i][j]->atom[n];
      if (atom_i == NULL){continue;}
      type_i = atom_i->type ;

      for(q=0 ; q < _N_SHELLS_ ; q++){ // reset all previous configurations
       number[q] = 0; //----------------------------------------------------// puts the counting table of neighbours at 0
       if (atom_i->neighbour[q]  != NULL){delete atom_i->neighbour[q];}
       if (atom_i->Jij[0][q] != NULL){delete atom_i->Jij[0][q];}

       if (_N_Jij_SETS_ == 2){  if (atom_i->Jij[1][q] != NULL){delete atom_i->Jij[1][q];}}
       
       for(n2=0 ; n2 < _MAX_ATOMS_PER_SHELL_ ; n2 ++){
        neighbour_buffer [q][n2] = NULL;
	Jij_buffer[0][q][n2] = 0;
	if (_N_Jij_SETS_ == 2){Jij_buffer[1][q][n2] = 0;}
       }
       
      } // Here, all previous configuration has ben erased
     
      
      //4 loops to seek in the neighbourhood of atom_i
      for(ht=-_CELL_RANGE_X_;ht<=_CELL_RANGE_X_;ht++){     hh = h + ht;
      for(it=-_CELL_RANGE_Y_;it<=_CELL_RANGE_Y_;it++){     ii = i + it;
      for(jt=-_CELL_RANGE_Z_;jt<=_CELL_RANGE_Z_;jt++){     jj = j + jt;
      
      if (MPI_p == 1){ // If one processor is used, it doesn't use the ghost area method but the real periodic boundary conditions.
       hh =(hh - L_min_x + L_eff_x) % L_eff_x + L_min_x;
       ii =(ii - L_min_y + L_eff_y) % L_eff_y + L_min_y;
       jj =(jj - L_min_z + L_eff_z) % L_eff_z + L_min_z;
      }
      
       for(n2 = 0 ; n2 < _N_ATOMS_PER_CELL_ ; n2++){
       
        atom_j = cell[hh][ii][jj]->atom[n2];
	if (atom_j == NULL){continue;}      
        type_j = atom_j->type ;
	
        R_ij = position_of(atom_i) - position_of(atom_j); // relative real vector betwen atom_i and atom_j

        Jij = 0 ;
	Jij_2 = 0;
        num_shell = 0;
	

/////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////     SETTING UP OF Jij's     //////////////////////////////////
                       /*       R_ij is the real vector between atom i and atom j.
                          type_i and type_j are the atom type of atom i and atom j.   */

/* You can use the "bring2zone" function to reduce the number of vectors to set up, 
only if you have all the required symetries in your lattice. Otherwise, don't use it.*/
R_ij = bring2zone(R_ij); 

if (type_i == 0 && type_j == 0) {
// R_ij.print(); // --- Can be activated to print the  relative locations on screen
if (R_ij == vect_(0.500 , 0.500 , 0.000)) {num_shell =  1; Jij =  0.00000;} // this first neighbour line is needed for the cluster creation process
/*
else if (R_ij == vect_(1.000 , 0.000 , 0.000)) {num_shell =  2; Jij =  0.00000;}
else if (R_ij == vect_(1.000 , 0.500 , 0.500)) {num_shell =  3; Jij =  0.00000;}
else if (R_ij == vect_(1.000 , 1.000 , 0.000)) {num_shell =  4; Jij =  0.00000;}
else if (R_ij == vect_(1.500 , 0.500 , 0.000)) {num_shell =  5; Jij =  0.00000;}
else if (R_ij == vect_(1.000 , 1.000 , 1.000)) {num_shell =  6; Jij =  0.00000;}
else if (R_ij == vect_(1.500 , 1.000 , 0.500)) {num_shell =  7; Jij =  0.00000;}// caution
else if (R_ij == vect_(2.000 , 0.000 , 0.000)) {num_shell =  8; Jij =  0.00000;}
else if (R_ij == vect_(2.000 , 0.500 , 0.500)) {num_shell =  9; Jij =  0.00000;}
else if (R_ij == vect_(1.500 , 1.500 , 0.000)) {num_shell =  9; Jij =  0.00000;}
else if (R_ij == vect_(2.000 , 1.000 , 0.000)) {num_shell = 10; Jij =  0.00000;}
else if (R_ij == vect_(1.500 , 1.500 , 1.000)) {num_shell = 11; Jij =  0.00000;}
else if (R_ij == vect_(2.000 , 1.000 , 1.000)) {num_shell = 12; Jij =  0.00000;}
*/
}
else if ((type_i == 0 && type_j == 1)||(type_i == 1 && type_j == 0)) {
/*
if (R_ij == vect_(0.500 , 0.500 , 0.000)) {num_shell =  1; Jij =  0.00000;}
else if (R_ij == vect_(1.000 , 0.000 , 0.000)) {num_shell =  2; Jij =  0.00000;}
else if (R_ij == vect_(1.000 , 0.500 , 0.500)) {num_shell =  3; Jij =  0.00000;}
else if (R_ij == vect_(1.000 , 1.000 , 0.000)) {num_shell =  4; Jij =  0.00000;}
else if (R_ij == vect_(1.500 , 0.500 , 0.000)) {num_shell =  5; Jij =  0.00000;}
else if (R_ij == vect_(1.000 , 1.000 , 1.000)) {num_shell =  6; Jij =  0.00000;}
else if (R_ij == vect_(1.500 , 1.000 , 0.500)) {num_shell =  7; Jij =  0.00000;}// caution
else if (R_ij == vect_(2.000 , 0.000 , 0.000)) {num_shell =  8; Jij =  0.00000;}
else if (R_ij == vect_(2.000 , 0.500 , 0.500)) {num_shell =  9; Jij =  0.00000;}
else if (R_ij == vect_(1.500 , 1.500 , 0.000)) {num_shell =  9; Jij =  0.00000;}
else if (R_ij == vect_(2.000 , 1.000 , 0.000)) {num_shell = 10; Jij =  0.00000;}
else if (R_ij == vect_(1.500 , 1.500 , 1.000)) {num_shell = 11; Jij =  0.00000;}
else if (R_ij == vect_(2.000 , 1.000 , 1.000)) {num_shell = 12; Jij =  0.00000;}
*/
}

else if (type_i == 1 && type_j == 1) {
//if (R_ij == vect_(1.000 , 0.000 , 0.000)) {num_shell =  1; Jij =1.0 ;}

if (R_ij == vect_(0.500 , 0.500 , 0.000)) {num_shell =  1; Jij =1.96972     ;}
else if (R_ij == vect_(1.000 , 0.000 , 0.000)) {num_shell =  2; Jij = 0.04572  ;}
else if (R_ij == vect_(1.000 , 0.500 , 0.500)) {num_shell =  3; Jij =  0.26130     ;}
else if (R_ij == vect_(1.000 , 1.000 , 0.000)) {num_shell =  4; Jij =  0.41667     ;}
else if (R_ij == vect_(1.500 , 0.500 , 0.000)) {num_shell =  5; Jij = -0.00958    ;}
else if (R_ij == vect_(1.000 , 1.000 , 1.000)) {num_shell =  6; Jij =  0.07944     ;}
else if (R_ij == vect_(1.500 , 1.000 , 0.500)) {num_shell =  7; Jij =  0.07023       ;}// caution
else if (R_ij == vect_(2.000 , 0.000 , 0.000)) {num_shell =  8; Jij =-0.00552     ;}
else if (R_ij == vect_(2.000 , 0.500 , 0.500)) {num_shell =  9; Jij =-0.00432     ;}
else if (R_ij == vect_(1.500 , 1.500 , 0.000)) {num_shell =  10; Jij =  0.13618     ;}
else if (R_ij == vect_(2.000 , 1.000 , 0.000)) {num_shell = 11; Jij =-0.00091    ;}
else if (R_ij == vect_(1.500 , 1.500 , 1.000)) {num_shell = 12; Jij =  0.00939   ;}
else if (R_ij == vect_(2.000 , 1.000 , 1.000)) {num_shell = 13; Jij = 0.01173    ;}
else if (R_ij == vect_(2.500 , 0.500 , 0.000)) {num_shell = 14; Jij = -0.00053  ;}
else if (R_ij == vect_(2.000 , 1.500 , 0.500)) {num_shell = 15; Jij = 0.02126   ;}
else if (R_ij == vect_(2.500 , 1.000 , 0.500)) {num_shell = 16; Jij =-0.00007  ;}
else if (R_ij == vect_(2.000 , 2.000 , 0.000)) {num_shell = 17; Jij = 0.04239  ;}
else if (R_ij == vect_(2.000 , 1.500 , 1.500)) {num_shell = 18; Jij = -0.00756;}
else if (R_ij == vect_(2.500 , 1.500 , 0.000)) {num_shell = 19; Jij = -0.00047;}
else if (R_ij == vect_(3.000 , 0.000 , 0.000)) {num_shell = 20; Jij = -0.00018;}
else if (R_ij == vect_(2.000 , 2.000 , 1.000)) {num_shell = 21; Jij = -0.00180;}
else if (R_ij == vect_(2.500 , 1.500 , 1.000)) {num_shell = 22; Jij =  0.00210;}
else if (R_ij == vect_(3.000 , 0.500 , 0.500)) {num_shell = 23; Jij =  0.00011;}
else if (R_ij == vect_(3.000 , 1.000 , 0.000)) {num_shell = 24; Jij =  0.00031;}
}

else if ((type_i == 1 && type_j == 2) || (type_i == 2 && type_j == 1) ) {

if (R_ij == vect_(0.250 , 0.250 , 0.250)) {num_shell =  1; Jij = 0.23627;}
else if (R_ij == vect_(0.750 , 0.250 , 0.250)) {num_shell =  2; Jij = -0.00257;}
else if (R_ij == vect_(0.750 , 0.750 , 0.250)) {num_shell =  3; Jij = 0.01312 ;}
else if (R_ij == vect_(1.250 , 0.250 , 0.250)) {num_shell =  4; Jij =0.00284  ;}
else if (R_ij == vect_(0.750 , 0.750 , 0.750)) {num_shell =  5; Jij = 0.00284 ;}
else if (R_ij == vect_(1.250 , 0.750 , 0.250)) {num_shell =  6; Jij = -0.00022;}
else if (R_ij == vect_(1.250 , 0.750 , 0.750)) {num_shell =  7; Jij = 0.00005  ;}
else if (R_ij == vect_(1.250 , 1.250 , 0.250)) {num_shell =  8; Jij = 0.00044  ;}
else if (R_ij == vect_(1.750 , 0.250 , 0.250)) {num_shell =  9; Jij = -0.00007 ;}
else if (R_ij == vect_(1.250 , 1.250 , 0.750)) {num_shell =  9; Jij =  -0.00015;}
else if (R_ij == vect_(1.750 , 0.750 , 0.250)) {num_shell = 10; Jij =0.00004   ;}
else if (R_ij == vect_(1.750 , 0.750 , 0.750)) {num_shell = 11; Jij = 0.00022  ;}
else if (R_ij == vect_(1.750 , 1.250 , 0.250)) {num_shell = 12; Jij =  -0.00006;}
else if (R_ij == vect_(1.250 , 1.250 , 1.250)) {num_shell = 13; Jij =-0.00014  ;}
else if (R_ij == vect_(1.750 , 1.250 , 0.750)) {num_shell = 14; Jij = -0.00002 ;}
else if (R_ij == vect_(2.250 , 0.250 , 0.250)) {num_shell = 15; Jij = -0.00001 ;}
else if (R_ij == vect_(2.250 , 0.750 , 0.250)) {num_shell = 15; Jij =  -0.00005;}
}

else if (type_i == 2 && type_j == 2) {

if (R_ij == vect_(0.500 , 0.500 , 0.000)) {num_shell =  1; Jij = 0.00310;}
else if (R_ij == vect_(1.000 , 0.000 , 0.000)) {num_shell =  2; Jij = -0.00051;}
else if (R_ij == vect_(1.000 , 0.500 , 0.500)) {num_shell =  3; Jij =  0.00007 ;}
else if (R_ij == vect_(1.000 , 1.000 , 0.000)) {num_shell =  4; Jij = 0.00018  ;}
else if (R_ij == vect_(1.500 , 0.500 , 0.000)) {num_shell =  5; Jij = -0.00003 ;}
else if (R_ij == vect_(1.000 , 1.000 , 1.000)) {num_shell =  6; Jij = -0.00001 ;}
else if (R_ij == vect_(1.500 , 1.000 , 0.500)) {num_shell =  7; Jij =  0.00000 ;}
else if (R_ij == vect_(2.000 , 0.000 , 0.000)) {num_shell =  8; Jij = 0.00002;}
else if (R_ij == vect_(2.000 , 0.500 , 0.500)) {num_shell =  9; Jij = -0.00001;}
}

/////////////////////////////////////////////////////////////////////////////////////////////////


if (num_shell != 0){
 num_shell--;
  neighbour_buffer[num_shell][number[num_shell]] = atom_j;
 Jij_buffer[0][num_shell][number[num_shell]] =  Jij * _Jij_SCALING_FACTOR_; // ferromag Jij's
 if (_N_Jij_SETS_ == 2){
  Jij_buffer[1][num_shell][number[num_shell]] = Jij_2 * _Jij_SCALING_FACTOR_; // paramag Jij's
 }
number[num_shell]++;
}

        }
       }}}
       
       // creates the neighbours links and associated Jij's of atom_i
       for (q2 = 0 ; q2 < _N_SHELLS_ ; q2++){
        
	if (number[q2] != 0){atom_i->Jij[0][q2] = new double[number[q2]];}
	else{atom_i->Jij[0][q2] = NULL;}
	
	atom_i->neighbour[q2] = new Atom*[(number[q2]+1)]; //paranteser borttagna

	if (_N_Jij_SETS_ == 2){if (number[q2] != 0){atom_i->Jij[1][q2] = new double[number[q2]];}
	else{atom_i->Jij[1][q2] = NULL;}}

	for (n2 = 0 ; n2 < number[q2] ; n2++){
	 atom_i->Jij[0][q2][n2] = Jij_buffer[0][q2][n2];
	 atom_i->neighbour[q2][n2] = neighbour_buffer[q2][n2];
	 
	 if (_N_Jij_SETS_ == 2){atom_i->Jij[1][q2][n2] = Jij_buffer[1][q2][n2];}
	}
	atom_i->neighbour[q2][number[q2]] = NULL; // the last item of the neighbour link list is a NULL pointer
	
       }
/*
       if(MPI_my_rank == 0){  // ------- uncomment this if you want to print on screen the number of neighbours of the atom # 0 of the Cell of coordinates [0,0,0]
        if (h==L_min_x && i==L_min_y && j==L_min_z && n == 0){
            for(q=0 ; q < _N_SHELLS_ ; q++){
            cout << " Shell number " << q+1 << " contains " << number[q] << " atoms." << endl;
            }
        }
       }
*/
     }
    }} if (MPI_my_rank == 0){cout << "\t" << (h+1-L_min_x) <<" of " << L_eff_x << endl;}
    }
    if (MPI_my_rank == 0){cout << "\t" << "\t" << "... Neighbours linked, Jij's set up." << endl <<endl ;}



 }




 void clear_mem(){     // Deletes cells, atoms, and clusters
    int h,i,j, s;
    extern int MPI_my_rank;

    for(h=0;h<L_tot_x;h++){
    for(i=0;i<L_tot_y;i++){
    for(j=0;j<L_tot_z;j++){
    if (cell[h][i][j] != NULL){delete(cell[h][i][j]);}
    }}}

    for(s = _MAX_CLUSTER_SIZE_ - 1 ; s <= 0 ; s--) {
     if (cluster[s] != NULL) {delete(cluster[s]) ;}
    }
    
    if (magnetic_atom != NULL){delete magnetic_atom;}
    
    free(MPI_Buffer);
    delete sub1;    delete sub2;    delete sub3;    delete sub4;    delete sub5;    delete sub6;
    delete ghost_sub1;    delete ghost_sub2;    delete ghost_sub3;    delete ghost_sub4;    delete ghost_sub5;    delete ghost_sub6;
    delete sub_line1;    delete sub_line2;    delete sub_line3;    delete sub_line4;    delete sub_line5;    delete sub_line6;    delete sub_line7;    delete sub_line8;    delete sub_line9;    delete sub_line10;    delete sub_line11;    delete sub_line12;
    delete ghost_sub_line1;    delete ghost_sub_line2;    delete ghost_sub_line3;    delete ghost_sub_line4;    delete ghost_sub_line5;    delete ghost_sub_line6;    delete ghost_sub_line7;    delete ghost_sub_line8;    delete ghost_sub_line9;    delete ghost_sub_line10;    delete ghost_sub_line11;    delete ghost_sub_line12;
    delete sub_corner1;    delete sub_corner2;    delete sub_corner3;    delete sub_corner4;    delete sub_corner5;    delete sub_corner6;    delete sub_corner7;    delete sub_corner8;
    delete ghost_sub_corner1;    delete ghost_sub_corner2;    delete ghost_sub_corner3;    delete ghost_sub_corner4;    delete ghost_sub_corner5;    delete ghost_sub_corner6;    delete ghost_sub_corner7;    delete ghost_sub_corner8;
    
    if(MPI_my_rank == 0){cout << endl <<"Memory cleanned." << endl ;}
 }



 inline void measure_magnetization(){
    int h, i, j, n, q, n2, Jij_set;

    mag = vect_(0,0,0);
    nrj_tot = 0;
    if (_USE_ONLY_MAGNETIC_ATOMS_ == 1){
     for(n=0 ; n<magnetic_population;n++){
      mag += magnetic_atom[n].link->m;
     }
     mag = mag * momentum[1] / magnetic_population;
    }
    else{

     for(h=L_min_x;h<L_max_x;h++){
     for(i=L_min_y;i<L_max_y;i++){
     for(j=L_min_z;j<L_max_z;j++){
      for(n = 0 ; n<_N_ATOMS_PER_CELL_ ; n++){
       mag += momentum[cell[h][i][j]->atom[n]->type] * cell[h][i][j]->atom[n]->m;

// measures total energy
       if (_MEASURE_NRJ_ == 1 ){
        for (q=0 ; q<_N_SHELLS_ ; q++){
         n2=0;
         while(cell[h][i][j]->atom[n]->neighbour[q][n2] != NULL){
	  Jij_set = which_Jij_set(cell[h][i][j]->atom[n]->neighbour[q][n2] ->m, cell[h][i][j]->atom[n]->m);
          nrj_tot += -cell[h][i][j]->atom[n]->Jij[Jij_set][q][n2]*(cell[h][i][j]->atom[n]->m * cell[h][i][j]->atom[n]->neighbour[q][n2]->m);
          n2++;
         }
        }
        }
       }
      }}}
      mag /= population;

    }

    // Collecting datas on all processors, the final result is stored on processor # 0
    m_tot_vect[0] = mag.x;
    m_tot_vect[1] = mag.y;
    m_tot_vect[2] = mag.z;

    MPI_Reduce(&m_tot_vect[0],&m_tot_vect_glob[0],3, MPI_DOUBLE, MPI_SUM, 0, MPI_Comm3D);
    MPI_Reduce(&nrj_tot,&nrj_tot_glob,1, MPI_DOUBLE, MPI_SUM, 0, MPI_Comm3D);

    if (MPI_my_rank == 0) {
     m_tot_glob = sqrt(m_tot_vect_glob[0]*m_tot_vect_glob[0]+m_tot_vect_glob[1]*m_tot_vect_glob[1]+m_tot_vect_glob[2]*m_tot_vect_glob[2]) / MPI_p;
     m_tot2_glob = m_tot_glob * m_tot_glob;
     m_tot4_glob = m_tot2_glob * m_tot2_glob;
   }

}


 inline void measure_angle_deviation(){ // RQ : total magnetisation vector mag has to be calculated before use of this function
    int h, i, j, n ;
    vect unit_mag = mag / norm_of(mag);
    
    for(h=L_min_x;h<L_max_x;h++){
    for(i=L_min_y;i<L_max_y;i++){
    for(j=L_min_z;j<L_max_z;j++){
      for(n = 0 ; n<_N_ATOMS_PER_CELL_ ; n++){
        angle_deviation_tot += acos(unit_mag * cell[h][i][j]->atom[n]->m);
       }
    }}}
      
    angle_deviation_tot /= population; 
    MPI_Reduce(&angle_deviation_tot,&angle_deviation_tot_glob,1, MPI_DOUBLE, MPI_SUM, 0, MPI_Comm3D);
    angle_deviation_tot_glob /= MPI_p;
    // the result is stored in angle_deviation_tot_glob, on processor # 0
}


//returns the real cartesian position of an atom within the sub-lattice (takes into account the basis vector)
 vect position_of(Atom *atom_i){ 
 Cell *cell_i = static_cast<Cell*>(atom_i->cell);
 vect buff = cell_i->pos + atom_i->pos;
 return vect_(buff * basis_vector[0],buff * basis_vector[1],buff * basis_vector[2]);
 }
 
 
 //returns the real distance between two atoms within the sub-lattice (takes into account the basis vector)
 double distance_between(Atom *atom_i, Atom *atom_j){
    return norm_of(position_of(atom_i) - position_of(atom_j));
}


// returns the distance (in Cells) to the closest sub-lattice border
 double how_far_from_border_is(Atom *my_atom){ 
      double d= _L_;
      int ax,ay,az;
      vect where;
      Cell *my_cell = static_cast<Cell*>(my_atom->cell);
      where = my_cell->pos + my_atom->pos;
      for(ax=0;ax<=1;ax++){
       d = min(d,abs(where.x - L_eff_x*ax));
      for(ay=0;ay<=1;ay++){
       d = min(d,abs(where.y - L_eff_y*ay));
      for(az=0;az<=1;az++){
       d = min(d,abs(where.z - L_eff_z*az));
      }}}
      return d;
 }
 
 
inline void measure_energy(){
    int h, i, j, n, q, n2, Jij_set;
    nrj_tot = 0;

     for(h=L_min_x;h<L_max_x;h++){
     for(i=L_min_y;i<L_max_y;i++){
     for(j=L_min_z;j<L_max_z;j++){
      for(n = 0 ; n<_N_ATOMS_PER_CELL_ ; n++){
        for (q=0 ; q<_N_SHELLS_ ; q++){
         n2=0;
         while(cell[h][i][j]->atom[n]->neighbour[q][n2] != NULL){
	  Jij_set = which_Jij_set(cell[h][i][j]->atom[n]->neighbour[q][n2] ->m, cell[h][i][j]->atom[n]->m);
          nrj_tot += -cell[h][i][j]->atom[n]->Jij[Jij_set][q][n2]*(cell[h][i][j]->atom[n]->m * cell[h][i][j]->atom[n]->neighbour[q][n2]->m);
          n2++;
         }
        }
       }
      }}}

    MPI_Reduce(&nrj_tot,&nrj_tot_glob,1, MPI_DOUBLE, MPI_SUM, 0, MPI_Comm3D);
}



 inline void measure_correlation(Atom * atom_i, Atom * atom_j, Statistic &correlation){
    if (atom_i != NULL && atom_j != NULL){
     correlation.add(atom_i->m * atom_j->m);
    }
 }



 void draw(double temperature, ofstream *out_file){ //prints the configuration of the lattice to file
    int h, i, j, n, current_type;
    vect local_mag, current_pos, current_m;

   if(out_file) {
   *out_file << "\t" << temperature << "\n" << "\n" ;
   *out_file << population << "\n" << "\n";

    for(h=L_min_x;h<L_max_x;h++){
    for(i=L_min_y;i<L_max_y;i++){
    for(j=L_min_z;j<L_max_z;j++){
     for(n = 0 ; n<_N_ATOMS_PER_CELL_ ; n++){
      current_pos = position_of(cell[h][i][j]->atom[n]);
      current_m = cell[h][i][j]->atom[n]->m;
      current_type = cell[h][i][j]->atom[n]->type;
      *out_file << current_pos.x << "_" << current_pos.y <<"_" << current_pos.z <<"_" << current_m.x << "_" << current_m.y << "_" << current_m.z <<"_" << current_type<< "_" << "\t \n";
      }
    }}}
    out_file->flush();
   }
 }

 };


 void Initialize_parralel_structure(MPI_Comm * MPI_Comm3D){
   int h, i, j;

   int ndim = 3;
   int dim_size[3] = {_NPC_X_,_NPC_Y_,_NPC_Z_};
   int pbc[3] = {1,1,1}; //periodic boundary conditions
   int coordinates[3];
   int coordi[3];

   extern int MPI_my_rank;
   extern int MPI_p;
   extern int neighbour_rank[3][3][3];

   MPI_Comm_size(MPI_COMM_WORLD, &MPI_p);

   MPI_Cart_create(MPI_COMM_WORLD, ndim, dim_size, pbc,1, MPI_Comm3D);
   MPI_Comm_rank(*MPI_Comm3D, &MPI_my_rank);
   MPI_Cart_coords(*MPI_Comm3D, MPI_my_rank, ndim ,coordinates);

   for (h = 0 ; h < 3 ; h++){
   for (i = 0 ; i < 3 ; i++){
   for (j = 0 ; j < 3 ; j++){
    coordi[0] = coordinates[0] + h - 1;
    coordi[1] = coordinates[1] + i - 1;
    coordi[2] = coordinates[2] + j - 1;
    MPI_Cart_rank(*MPI_Comm3D, coordi, &neighbour_rank[h][i][j]);

   }}}

   cout << "Hello from process " << MPI_my_rank << endl;
   MPI_Barrier(*MPI_Comm3D);
}

////// --- DECLARATIONS ---- //////
void MPI_my_init(int argc, char** argv);
void MonteCarlo(Lattice &lattice);
void save_datas(double temperature, unsigned long t, double mag , double mag2, double mag4, double nrj,  ofstream *out_file);
void signature(ofstream *out_file);
void save_data_in_file(double data, ofstream *out_file, boolean return_carriage=yes);
void metropolis(Atom*atom, double temperature);
void constrained_metropolis(Atom*atom, double temperature);
void heat_bath(Atom* atom, double temperature);
int cluster_metropolis(Atom *kernel, double temperature, int pop);
int wolff(Atom *kernel, double temperature);
vect rotate_vect(vect old_spin, vect M,double angle = -137);
////// ---              ---- //////


/////////////////////////////////////////////////////////////////////////////
//////////////////////////// ----   MAIN    ---- ////////////////////////////
/////////////////////////////////////////////////////////////////////////////


int main(int argc, char** argv) {
   extern int MPI_my_rank;
   MPI_Init(&argc, &argv);
   MPI_Comm MPI_Comm3D;
   Initialize_parralel_structure(&MPI_Comm3D);

   if (MPI_my_rank == 0){cout << endl << endl << "Starting..." << endl << endl;}

   register Lattice lattice(MPI_Comm3D);
   MonteCarlo(lattice);

   if (MPI_my_rank == 0){cout << endl <<"... End" << endl;}

   MPI_Finalize();
}



/////////////////////////////////////////////////////////////////////////////
//////////////////////////// ---- MONTECARLO---- ////////////////////////////
/////////////////////////////////////////////////////////////////////////////

void MonteCarlo(Lattice &lattice) {
   boolean good_average, need2measure, need2break;
   int ll,mm,nn,number, population;
   extern int MPI_my_rank;
   unsigned long t, t_mc, t_inter_measure, tt, tt_mc, t_buff, which;
   double temperature, time_of_calc =0 ;
   double magnetization, susceptibility, cumulant, energy;
   vect old_spin, new_spin;
   Atom  *who;
   //Atom *atom_i= NULL, *atom_j= NULL, *atom_k= NULL, *atom_l= NULL;
   Statistic mag(_T_CONVERG_),  long_mag(_T_MEASURE_), long_mag2(_T_MEASURE_),long_mag4(_T_MEASURE_), long_nrj(_T_MEASURE_), correlation_intra(_T_MEASURE_),correlation_inter(_T_MEASURE_), angle_deviation(_T_MEASURE_);
   ofstream data_file, result_file, draw_file;

   MPI_Comm MPI_Comm3D = lattice.MPI_Comm3D;

   if (MPI_my_rank == 0){
    data_file.open("data.dat");
    result_file.open("result.dat");
    draw_file.open("draw.dat");
    time_of_calc = MPI_Wtime();
   }

/*
   if (_CLUSTERS_ == 1){
    lattice.cluster[4][0].atom[0].link; //SELECT THE TWO ATOMS TO CALCULATE INTER CLUSTER CORRELATION
    lattice.cluster[4][2].atom[3].link;
    lattice.cluster[4][0].atom[0].link; //SELECT THE TWO ATOMS TO CALCULATE INTRA CLUSTER CORRELATION
    lattice.cluster[4][0].atom[3].link;
    if (MPI_my_rank == 0){
     cout << "Distance between the 2 atoms on which inter-cluster correlation is calculated : " << lattice.distance_between(atom_i, atom_j ) << endl;
     cout << "Distance between the 2 atoms on which intra-cluster correlation is calculated : " << lattice.distance_between(atom_k, atom_l ) << endl;
    }
   }
*/

   srand(_SEED_);

   lattice.measure_magnetization();

   if(_USE_ONLY_MAGNETIC_ATOMS_ == 1){t_inter_measure = lattice.magnetic_population; population =lattice.magnetic_population;}
     else{t_inter_measure = lattice.population; population =lattice.population;}

   if (MPI_my_rank == 0){
    cout  <<endl << "-- MONTE CARLO PROCESS -- method n°"<<_METHOD_ << endl << endl ;
    cout << "Initial magnetization : "<<  lattice.m_tot_glob << endl;
   }

   for (temperature = _TEMP_MAX_;temperature >= _TEMP_MIN_;temperature -= _TEMP_STEP_) {  // --- LOOP OVER TEMPERATURE, going down
   //for (temperature = _TEMP_MIN_;temperature <= _TEMP_MAX_;temperature += _TEMP_STEP_) {  // --- LOOP OVER TEMPERATURE, going up
   
    if (MPI_my_rank == 0){cout  << endl <<"Temperature : " << temperature<< endl ;}

    t=0;
    t_mc=0;

    good_average = no;
    need2break = no;
                                     // resets all "Statistic" objects
    mag.clear();
    long_mag.clear();
    long_mag2.clear();
    long_mag4.clear();
    long_nrj.clear();
    correlation_intra.clear();
    correlation_inter.clear();
    angle_deviation.clear();

    MPI_Barrier(MPI_Comm3D);
    
    //----------------------------//    
  // CONVERGENCE PHASE //
//----------------------------//

    do{ // ------------------------------------------------------------------------------------- LOOP OVER TIME
     t++;
     need2measure = no;

     // Selection of the atom to be flipped
     if(_USE_ONLY_MAGNETIC_ATOMS_ == 1){
      which = random_range(0,lattice.magnetic_population) ;   //Chooses a spin among the magnetic atoms
      who = lattice.magnetic_atom[which].link;
     }
     else{
      ll = random_range(L_min_x, L_max_x) ; // Chooses a spin among all spins
      mm = random_range(L_min_y, L_max_y) ;
      nn = random_range(L_min_z, L_max_z) ;
      number = rand() % _N_ATOMS_PER_CELL_;
      who = lattice.cell[ll][mm][nn]->atom[number];
     }

     switch (_METHOD_) { // Performs the updating, with one of the following methods
      case 0:
       metropolis(who, temperature);
       break;
      
      case 1:
       if ((t_mc % 2 ) > 1){constrained_metropolis(who, temperature);}
       else{metropolis(who, temperature);}
       break;

       case 3:
        heat_bath(who, temperature);
        break;

       case 4:
        metropolis(who, temperature);
	break;

       case 5:
        if (t_mc < 1000 && temperature == _TEMP_MAX_){ // starting with a little bit of plain Metropolis
	//if (t_mc < 1000 && temperature == _TEMP_MIN_){
         metropolis(who, temperature); 
        }
        else{
         t_buff = t % t_inter_measure;
         t = (t + wolff(who,temperature) - 1) % t_inter_measure;
         if (t_buff >= t){need2measure = yes;}
        }
	//need2measure = does_any_processor(need2measure, &MPI_Comm3D);   
	break;
     }
     
     if (t % t_inter_measure ==  0 ){t = 0; need2measure = yes;} // Takes measurements at each MC sweep
              
     if (need2measure == yes){
      t_mc++; // t_mc reprensents the time in MC steps
      lattice.refresh_ghost_areas();
      lattice.measure_magnetization();

      if (MPI_my_rank == 0){
       mag.add(lattice.m_tot_glob);
       mag.calcul_average();

	if (t_mc > _T_MIN_ ){need2break = mag.is_good_average(_GOOD_AVERAGE_);}
	
       save_datas(temperature, t_mc, lattice.m_tot_glob, lattice.m_tot2_glob,  lattice.m_tot4_glob, lattice.nrj_tot_glob, &data_file);
      }
      
      need2break = does_any_processor(need2break ,&MPI_Comm3D);
    }


     }while(t_mc <= _T_CONVERG_ && need2break == no); // --- Continues time loop if still fluctuating

   if(MPI_my_rank == 0){cout << "Begining of measurement phase..." << endl ;}

   MPI_Barrier(MPI_Comm3D);
   good_average = no;
   need2break = no;
   tt_mc = 0;

    //-----------------------------//    
  // MEASUREMENT PHASE //
//-----------------------------//

do{ // ---- Loop over time
     tt++;
     need2measure = no;
     
     // Selection of the atom to be flipped
     if(_USE_ONLY_MAGNETIC_ATOMS_ == 1){
      which = random_range(0,lattice.magnetic_population) ;   //Chooses a spin among the magnetic atoms
      who = lattice.magnetic_atom[which].link;
     }else{
      ll = random_range(L_min_x, L_max_x) ;   //Chooses a spin among all spins
      mm = random_range(L_min_y, L_max_y) ;
      nn = random_range(L_min_z, L_max_z) ;
      number = rand() % _N_ATOMS_PER_CELL_; 
      who = lattice.cell[ll][mm][nn]->atom[number];
     }


     switch (_METHOD_) {    // Performs the updating, with one of the following methods
      case 0:
       metropolis(who, temperature);
       break;

      case 1:
       if ((tt_mc % 2 ) > 1){constrained_metropolis(who, temperature);}
       else{metropolis(who, temperature);}
       break;

      case 3:
        heat_bath(who, temperature);
        break;

      case 4:
        t_buff = tt % t_inter_measure;
        tt = (tt + cluster_metropolis(who,temperature,population) - 1)  % t_inter_measure ;
	if (t_buff >= tt){need2measure = yes;}
	need2measure = does_any_processor(need2measure, &MPI_Comm3D);   
	break;

       case 5:
        t_buff = tt % t_inter_measure;
        tt = (tt + wolff(who,temperature) - 1) % t_inter_measure;
	if (t_buff >= tt){need2measure = yes;}
	//need2measure = does_any_processor(need2measure, &MPI_Comm3D);   
	break;
     }
     
     if (tt % t_inter_measure ==  0 ){tt = 0; need2measure = yes;} // Takes measurements at each MC sweep
     
     if (need2measure == yes){   
      tt_mc++;  // tt_mc reprensents the time in MC steps
      if (tt_mc % _T_REFRESH_GHOST_AREAS_ == 0) {lattice.refresh_ghost_areas();}
      lattice.measure_magnetization();
/*   lattice.measure_angle_deviation();
      lattice.measure_correlation(atom_i, atom_j, *&correlation_inter);
      lattice.measure_correlation(atom_k, atom_l, *&correlation_intra);
*/
      if (MPI_my_rank == 0){
       if (tt_mc % _T_REFRESH_GHOST_AREAS_ == 0) {save_datas(temperature, t_mc + tt_mc, lattice.m_tot_glob, lattice.m_tot2_glob,  lattice.m_tot4_glob, lattice.nrj_tot_glob, &data_file);}

       long_mag.add(lattice.m_tot_glob);
       long_mag.calcul_average();
       if (tt_mc > _T_MIN_ ){
       good_average = long_mag.is_good_average(_GOOD_AVERAGE_);
       if (good_average == yes) {need2break = yes;} // Break the loop if the average is good enough
       }
       long_mag2.add(lattice.m_tot2_glob);
       long_mag4.add(lattice.m_tot4_glob);
       long_nrj.add(lattice.nrj_tot_glob);
       //angle_deviation.add(lattice.angle_deviation_tot_glob);

      }

      need2break = does_any_processor(need2break ,&MPI_Comm3D);
     }


    }while(tt_mc < _T_MEASURE_ && need2break==no );


    if (MPI_my_rank == 0){ // Evaluates final averages
     long_mag.calcul_average();
     long_mag2.calcul_average();
     long_mag4.calcul_average();
     long_nrj.calcul_average();
/*  correlation_intra.calcul_average();
     correlation_inter.calcul_average();
     cout << "CORRELATION INTER CLUSTER : " << correlation_inter.average <<  endl ;
     cout << "CORRELATION INTRA CLUSTER : " << correlation_intra.average <<  endl ;
     angle_deviation.calcul_average();
     cout << "\t" << "Mean angular deviation of spins = " << "\t" <<angle_deviation.average << endl;
 */

     // PHYSICAL MEASUREMENTS
     magnetization = long_mag.average;
     susceptibility = population * (long_mag2.average - long_mag.average*long_mag.average) / (_k_ * (temperature+0.0001));
     cumulant = 1.0 - long_mag4.average/(3.0*long_mag2.average*long_mag2.average);
     energy = long_nrj.average;

     save_datas(temperature, t_mc + tt_mc, magnetization ,susceptibility ,cumulant, energy, &result_file);

     if (need2break == no){cout << "Bad average... break." << endl ;}
     cout << "\t" << "\t" <<"... End of Measurement phase." << endl ;
     cout << "   Time : " << t_mc + tt_mc  << "\t" << "Mean mag : " << magnetization << "\t Suscept. : " << susceptibility << "\t" << "4th B. Cumulant : " <<  cumulant << "\t"<< "Total energy : " << energy << endl;

     //lattice.put_spins_up();   //----------------------------------------------------------------puts spins up & aligned for the next temperature step
     //if (temperature == 55 || temperature == 35 || temperature <= 20){lattice.draw(temperature, &draw_file);} // prints the lattice to file
    }


   }
   MPI_Barrier(MPI_Comm3D);

   if (MPI_my_rank == 0){
    data_file.close();
    result_file.close();
    draw_file.close();
    remove("data.dat"); // removes "data.dat" at the end of the process
    
    data_file.open("time.dat");
    time_of_calc = MPI_Wtime() - time_of_calc ;
    signature(&data_file);
    data_file << "Time of calculation : " << "\t "<< trunc(time_of_calc/3600,2) << " hours.";
    data_file.flush();
    data_file.close();
   }

}


inline void save_datas(double temperature, unsigned long t, double mag , double mag2, double mag4, double nrj,ofstream *out_file){
   if(out_file) {
    *out_file << "\t" << temperature << "\t" << t << "\t" << mag << "\t" << mag2 << "\t" << mag4 << "\t" << nrj ;
    *out_file << "\n";
     out_file->flush();  
   }
}


void signature(ofstream *out_file){
   if(out_file) {
    *out_file << "General Monte Carlo Program" << "\n" ;
    *out_file << "written by Camille Aron," << "\n" ;
    *out_file << "Theoretical Magnetism Group -- Ångström Laboratory -- Uppsala University -- Sweden," << "\n" ;
    *out_file << "Dép. de physique --- École Normale Supérieure de Cachan --- France." << "\n" << "\n" ;
    out_file->flush();  
   }
}


inline void save_data_in_file(double data, ofstream *out_file, boolean return_carriage){ //print data to file
   if(out_file) {
    *out_file << "\t" << data ;
    if (return_carriage == yes){*out_file << "\n";}
     out_file->flush();  
   }
}


/*"my_answer" is looked at, on each processor.
If one of them is equal to "yes" then, this function will return "yes" on every processor*/
inline boolean does_any_processor(boolean my_answer, MPI_Comm * Comm){
   int int_my_answer, answer;
   
   MPI_Barrier(*Comm);
   if (my_answer == yes){int_my_answer = 1;}else{int_my_answer = 0;}
   MPI_Reduce(&int_my_answer, &answer, 1, MPI_INT, MPI_SUM, 0, *Comm);
   MPI_Bcast(&answer,1,MPI_INT,0,*Comm);
   if (answer > 0){return yes;}else{return no;}
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////// ---- METROPOLIS---- ////////////////////////////
//////////////////////////////////////////////////////////////////////////////
inline void metropolis(Atom*atom, double temperature){
   double de, p;
   vect new_spin ;
   new_spin = new_random_spin();
   atom->get_local_mag();
   de = atom->local_mag * (atom->m - new_spin); // --- de = energy difference
   p = min(1.0,exp(-de/(_k_ * temperature))); 
   if(random_unit() <= p){atom->flip(new_spin);}
/* // if you want to follow the acceptance rate
   static Statistic watch_p(1000);
   watch_p.add(p);
   watch_p.calcul_average();
   if(watch_p.current == 0){cout << watch_p.average <<  endl;}
*/
} 


/*Not tested completely*/
inline void constrained_metropolis(Atom* atom, double temperature){
   double nrj_before, de, draw, p;
   vect old_spin ;

   old_spin = atom->m;
   nrj_before =atom->nrj_neighbour();

   if (atom->local_mag == vect_(0,0,0) ){
    atom->flip(new_random_spin());
   }
   else{
    atom->flip(rotate_vect(old_spin, atom->local_mag));
   }

   de = atom->nrj_neighbour() - nrj_before; // --- de = energy difference
   p = min(1.0,exp(-de/(_k_ * temperature))); //------------------------Standard Metropolis
   draw = random_unit();
   if(draw > p){atom->flip(old_spin);}

/* // if you want to follow the acceptance rate
  static Statistic watch_p(1000);
  watch_p.add(p);
  watch_p.calcul_average();
  if(watch_p.current == 0){cout << watch_p.average <<  endl;}
*/
}


/* Creates some cluster of atoms to be flipped  (by standard metropolis) regarding the aceptance rate
 large accpetance -> large clusters 
 meant to speed up the fluctuations and though get a better average qiucker....
 not sure at 100 %
 */
inline int cluster_metropolis(Atom *who, double temperature, int pop){
   boolean choice;
   int which,which_shell,q,n,i,j,N, occupancy[_N_SHELLS_],count_tries,count_tries2;
   double p,energy = 0,boltzman_factor;
   Atom *old_cluster_atom, *previous;
   Atomlink * cluster_atom;
   static int N_max=1;
   static Statistic watch_N(1000);

   N = random_range(1, N_max+1);
   old_cluster_atom = new Atom[N];
   cluster_atom = new Atomlink[N];

   for(i=0;i< N ; i++){

    old_cluster_atom[i] = *who;
    cluster_atom[i].link = who;
    energy -= cluster_atom[i].link->nrj_neighbour();
   
    if (i == N-1){break;} // no need to find another neighbour, the cluster is finished
   
// construction of the cluster (using the nearest neighbour array)
    for (q=0 ; q<_N_SHELLS_ ; q++){ // count the number of nearest neighbours in each shell
     n=0;
     while(who->neighbour[q][n] != NULL){n++;}
     occupancy[q] = n;
    }

     count_tries2 = 0;
     previous = who;
     do{
      count_tries2++;
      if (count_tries2 > 20){N=i+1;break;}
     
      count_tries=0;
      do{
       count_tries++;
       which_shell = random_range(0,_N_SHELLS_);
       if (count_tries > 20){break;}
      }while(occupancy[which_shell] == 0);
      if (count_tries > 20){N = i+1;break;}
         
      which = random_range(0,occupancy[which_shell]);
      who =previous->neighbour[which_shell][which];    

      choice = yes;
      for (j=0;j<=i;j++){if (who == cluster_atom[j].link){choice = no;}}
      if (who->is_in_real_area() == no){choice =no;}
     }while(choice==no);

   }
   
   // flipping the cluster at random
   for(i=0;i< N ; i++){
    cluster_atom[i].link->flip(new_random_spin());
   }
   
   // calculating the energy difference between the new and the old configuration
   for(i=0;i< N ; i++){
    energy +=cluster_atom[i].link->nrj_neighbour();
   }

   boltzman_factor = exp(-energy/(_k_ * temperature));
   watch_N.add(boltzman_factor); // making a statistic of the acceptance rate
   p = min(1.0,boltzman_factor); //------------------------Standard Metropolis

   if(random_unit() <= p){}  // move accepted
   else{                                 // rejected, going back to old configuration
    for(i=0;i< N ; i++){cluster_atom[i].link->flip(old_cluster_atom[i].m);}
   }

   if (watch_N.current == 999){ // reavaluating the optimal cluster size
    watch_N.calcul_average();
    N_max = int(min(double(pop), 1 + watch_N.average * pop));
   }

   delete cluster_atom;
   delete old_cluster_atom;

   return N; // return the size of the cluster used.
}


//////////////////////////////////////////////////////////////////////////////
//////////////////////////// ---- HEAT-BATH ---- ////////////////////////////
//////////////////////////////////////////////////////////////////////////////

inline void heat_bath(Atom* atom, double temperature){
   double P1, phi, J, theta, buff;
   vect old_spin, new_spin, local_mag;
  
   atom->get_local_mag();
   local_mag =atom->local_mag;
   J = norm_of(local_mag);
   
   if (J != 0){
    phi = 2.0 *Pi * random_unit();
    P1 = random_unit();
    
    buff = 1.0 + (_k_*temperature/J) * log(1.0 + P1*(exp(-2.0 * J / (_k_ * temperature))-1.0));
    
    // veryfying the range of the input of the following arc-cosinus function
    if (buff > 1.0){buff = 1.0;}
    else if (buff < -1.0){buff = -1.0;}
    
    theta = acos(buff);
     
    do {	// building a precise orthogonal vector to local_mag
     new_spin = vect_(1.0, random_unit(), random_unit()) ^local_mag;
    }while((norm_of(new_spin)< 0.05 * J));
    
    new_spin = rotate_vect(local_mag/J,new_spin,theta);//rotating local mag around this new vector
    new_spin = rotate_vect(new_spin, local_mag, phi);//rotating the resulting vector around local_mag
    atom->flip(new_spin);
   }
 
}


/////////////////////////////////////////////////
////////// WOLFF ALGORITHM /////////////////
/////////////////////////////////////////////////
/* Phys. Rev. Lett. 62, 361 (1989)  */
//should only be used with 1 processor

/*Object created for each atom of the cluster, it contains a link toward this atom : Atom * link.
In the end of the cluster creation, all these objects will form an "chain" of links toward the cluster atoms*/
struct atom_in_cluster{
   boolean all_neighbours_explored;
   Atom * link ;
   atom_in_cluster *prec, *next; 
   //prec is a link towards the last "atom_in_cluster" created
   //next is a link towards the future "atom_in_cluster" created
   //so for a atom_in_cluster object situated in the middle of the chain, it knows its two neighbours in the chain.

   atom_in_cluster(Atom * atom_link, atom_in_cluster *prec_link = NULL){
    // the first argument is (should be) a link towards the corresponding new cluster atom.
    // the second argument is (should be) a link towards the last "atom_in_cluster" created.
    //For the first "atom_in_cluster" created (the kernel of the cluster) no second argument is required
    link = atom_link;
    link->tag = yes;
    all_neighbours_explored = no;
    next = NULL;
    prec = NULL;
    if (prec_link != NULL){
     prec = prec_link;
     prec_link->next = this;
    }
   }
   
   
   ~atom_in_cluster(){ // deletes the whole cluster
   link->tag = no;
   if (this->next !=NULL){delete (this->next);}   
   }
  
  
};

inline int wolff(Atom *kernel_atom, double temperature){
   boolean in_cluster;
   int q,n,Jij_set, counter=1;
   double p;
   vect r, new_spin;
   Atom *who ;
   atom_in_cluster *current_atom_in_cluster, *kernel, *new_atom; // *current
      
   r = new_random_spin();
   who = kernel_atom;
   who->flip(who->m - 2.0 * (who->m * r) * r); // flips the first atom (kernel)
   kernel = new atom_in_cluster(who);
   current_atom_in_cluster = kernel;
   new_atom = kernel;

  do{ // construction of the cluster (using the nearest neighbour array of "who")
   
   who = current_atom_in_cluster->link;
   
   if (current_atom_in_cluster->all_neighbours_explored == no && who->is_in_real_area()==yes){

    for (q=0 ; q<_N_SHELLS_ ; q++){ // seeks all the neighbours of "who"
     n=0;
     while(who->neighbour[q][n] != NULL){
     
     if (_USE_ONLY_MAGNETIC_ATOMS_ == 1){if (who->neighbour[q][n]->type != 1){n++;continue;}}
     if (who->neighbour[q][n]->is_in_real_area()==no){n++;continue;}
   
     // checks if the current neighbour is already in the cluster or not
     if (who->neighbour[q][n]->tag==yes){in_cluster = yes;}else{in_cluster = no;}
/*    
	in_cluster = no;
        current = new_atom;
        do{
         if (current->link == who->neighbour[q][n]){in_cluster = yes;break;}
         current = current ->prec;
        }while(current != NULL);
*/	
     // checking finished, result is stored in choice
     
     if (in_cluster == no) {
      Jij_set = which_Jij_set(who->m, who->neighbour[q][n]->m);
      p = 1.0 - exp(min(0.0, 2.0 * who->Jij[Jij_set][q][n] * (r*who->m) * (r*who->neighbour[q][n]->m) / (_k_ * temperature)));
      
      if (random_unit() <= p){   //current neighbour accepted in the cluster
       who->neighbour[q][n]->flip(who->neighbour[q][n]->m - 2.0 * (who->neighbour[q][n]->m * r) * r);
       new_atom = new atom_in_cluster(who->neighbour[q][n], new_atom);
       counter++;
      }
    
     }
     n++;
     }
    }
    current_atom_in_cluster->all_neighbours_explored =yes ;   
   }  
   current_atom_in_cluster = current_atom_in_cluster->next;
   }while(current_atom_in_cluster != NULL);
    
   delete(kernel); // the whole cluster object is deleted here
   return counter; // returns the size of the cluster
}

// Determines which set of Jij should be used between 2 spins according to the 2 spins
inline int which_Jij_set(vect m1, vect m2){
return 0; //if only one set is used (most common) you should put this line
//if ((m1*m2) >= _cos_limit_angle_){return 0;}else{return 1;}; // else put this one
}


/////////////////////////////////////////////////////////////////////////////
//////////////////////////// ----  ALGEBRA  ---- ////////////////////////////
/////////////////////////////////////////////////////////////////////////////


//returns a random unit vector.
/* if this algorithm doesn't seem right to you...think about it again, 
it's damned the best way of picking a fully random spin (with flat distribution) ever !
    -- recommanded by Binder himself...*/
inline vect new_random_spin(){
   double x, y, z, r=1.0;
   vect new_spin;
   
   while(r > 0.5 || r == 0){
   x = random_unit() - 0.5 ;
   y = random_unit() - 0.5 ;
   z = random_unit() - 0.5 ;
   r = sqrt(x*x + y*y + z*z);
   }
   
   new_spin.x = x / r;
   new_spin.y = y / r;
   new_spin.z = z / r;

   return new_spin;
}

// rotates "old_spin" around "M" of  "theta" radians
inline vect rotate_vect(vect old_spin, vect M, double theta){ 
   matrix mat_rotation;
   
   M /= norm_of(M);
   
   if (theta == -137){
    theta =  Pi + 0.2 - 0.4 * random_unit();
   }
   
   mat_rotation.a[0][0] = M.x*M.x + (M.y*M.y + M.z*M.z) * cos(theta);
   mat_rotation.a[0][1] = M.x * M.y * (1-cos(theta)) - M.z * sin(theta);
   mat_rotation.a[0][2] = M.x * M.z * (1-cos(theta)) + M.y * sin(theta);

   mat_rotation.a[1][0] = M.x * M.y * (1-cos(theta)) + M.z * sin(theta);
   mat_rotation.a[1][1] = M.y*M.y + (M.x*M.x + M.z*M.z) * cos(theta);
   mat_rotation.a[1][2] = M.y * M.z * (1-cos(theta)) - M.x * sin(theta);

   mat_rotation.a[2][0] = M.x * M.z * (1-cos(theta)) - M.y * sin(theta);
   mat_rotation.a[2][1] = M.y * M.z * (1-cos(theta)) + M.x * sin(theta);
   mat_rotation.a[2][2] = M.z*M.z + (M.x*M.x + M.y*M.y) * cos(theta);

   return mat_rotation * old_spin;
}

coord coord_(int l, int m, int n){
   coord tamp;
   tamp.l = l;
   tamp.m = m;
   tamp.n = n;
   return tamp;
}

inline vect vect_(double x, double y, double z){
   vect tamp;
   tamp.x = x;
   tamp.y = y;
   tamp.z = z;
   return tamp;
}

inline coord operator - (coord c1, coord c2){
   coord tamp;
   tamp.l = c1.l - c2.l;
   tamp.m = c1.m - c2.m;
   tamp.n = c1.n - c2.n;
   return tamp;
}

inline vect operator + (vect v1, vect v2){
   vect tamp;
   tamp.x = v1.x + v2.x;
   tamp.y = v1.y + v2.y;
   tamp.z = v1.z + v2.z;
   return tamp;
}

inline vect operator - (vect v1, vect v2){
   vect tamp;
   tamp.x = v1.x - v2.x;
   tamp.y = v1.y - v2.y;
   tamp.z = v1.z - v2.z;
   return tamp;
}

// Scalar product
inline double operator * (vect v1, vect v2){
   return (v1.x*v2.x + v1.y*v2.y + v1.z*v2.z);
}

inline vect operator + (coord c1, vect v2){
   vect tamp;
   tamp.x = c1.l + v2.x;
   tamp.y = c1.m + v2.y;
   tamp.z = c1.n + v2.z;
   return tamp;
}

inline vect operator - (coord c1, vect v2){
   vect tamp;
   tamp.x = c1.l - v2.x;
   tamp.y = c1.m - v2.y;
   tamp.z = c1.n - v2.z;
   return tamp;
}

inline vect operator * (vect v, double lambda){
   vect tamp;
   tamp.x = lambda * v.x;
   tamp.y = lambda * v.y;
   tamp.z = lambda * v.z;
   return tamp;
}

//vectorial product
inline vect operator ^ (vect v1, vect v2){ 
   vect tamp;
   tamp.x = v1.y * v2.z - v1.z * v2.y;
   tamp.y = v2.x * v1.z - v1.x * v2.z;
   tamp.z = v1.x * v2.y - v1.y * v2.x;
   return tamp;
}
inline vect operator * (double lambda,vect v){
   vect tamp;
   tamp.x = lambda * v.x;
   tamp.y = lambda * v.y;
   tamp.z = lambda * v.z;
   return tamp;
}

inline vect operator / (vect v, double lambda){
   vect tamp;
   if (lambda != 0){
    tamp.x =  v.x / lambda;
    tamp.y = v.y / lambda;
    tamp.z = v.z / lambda;
   return tamp;
   }
   else{ cout << "Pb of division by zero in vect operator /" << endl; return v;}
}

inline int operator==(vect v1, vect v2){
   if (v1.x == v2.x && v1.y == v2.y && v1.z == v2.z ){return 1;}else{return 0;}
}

// Multiplication of a vector by a matrix
inline vect operator * (matrix mat, vect v){
   int i, j;
   double vv[3], tt[3] = {0,0,0};

   vv[0] = v.x;
   vv[1] = v.y;
   vv[2] = v.z;

   for (i=0; i<3 ; i++){
   for (j=0; j<3 ; j++){
    tt[i] += mat.a[i][j] * vv[j] ;
   }}

   return vect_(tt[0],tt[1],tt[2]);
}

//returns the norm of a vector
inline double norm_of(vect one){
   return (double(sqrt(one.x * one.x + one.y * one.y + one.z * one.z)));
}

//Coordinates of the resulting vector will be positive, classified in decreasing order.
vect bring2zone(vect one){
   double buffer ;
   vect tamp;
   
   tamp.x = abs(one.x);
   tamp.y = abs(one.y);
   tamp.z = abs(one.z);

   if (tamp.x < tamp.y){buffer = tamp.x ; tamp.x = tamp.y ; tamp.y = buffer ;}
   if (tamp.y < tamp.z){buffer = tamp.y ; tamp.y = tamp.z ; tamp.z = buffer ;}
   if (tamp.x < tamp.y){buffer = tamp.x ; tamp.x = tamp.y ; tamp.y = buffer ;}

   return tamp;
}

// returns the unit vector corresponding to theta and phi Euler angle
inline vect getvect(double theta,double phi){
   vect vector;
   vector.x = cos(phi) * sin(theta);
   vector.y = sin(phi) * sin(theta);
   vector.z = cos(theta);
   return vector ;
}

// returns an integer type random number not including the highest number
inline int random_range(int lowest_number, int highest_number){   
    int range = highest_number - lowest_number;
    return lowest_number + int(range * double(rand())/(RAND_MAX + 1.0));
}

// returns a double type random number between 0 and 1, including 0 and 1.0
inline double random_unit(){ 
  return double(rand())/(RAND_MAX);  
}

inline double trunc(double value, int order){
 double buff = pow(10,order);
 return double(int(value * buff))/buff ;
}

inline double min(double x, double y){
   if (x<y){return x;}else{return y;}
}
inline double max(double x, double y){
   if (x>y){return x;}else{return y;}
}

inline double abs(double x){
   if (x > 0) {return x;} else {return -x;}
}

