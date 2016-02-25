namespace sierra {
namespace nalu {

  static constexpr int triangularFacetTable[4][3] = {
    {4, 0, 1},
    {4, 1, 2},
    {4, 2, 3},
    {4, 3, 0}};
void inline
quadAreaByTriangleFacets(const double  areacoords[4][3], const SharedMemView<double*[3][16]>& area,const int ics,const int v)
{
  /*constexpr int triangularFacetTable[4][3] = {
    {4, 0, 1},
    {4, 1, 2},
    {4, 2, 3},
    {4, 3, 0}};
*/
  alignas(64) double xmid[3];
  alignas(64) double r2[3];
  for(int k=0; k < 3; ++k)
  {
    xmid[k] = 0.25 * (areacoords[0][k] + areacoords[1][k] + areacoords[2][k] + areacoords[3][k]);
    area(ics,k,v) = 0.;
    r2[k] = areacoords[0][k] - xmid[k];
  }
  const int ntriangles = 4;
  alignas(64) double r1[3];
  for(int itriangle = 0; itriangle < ntriangles; ++itriangle)
  {
    const int iq = triangularFacetTable[itriangle][2];
    for(int k=0; k < 3; ++k)
    {
      r1[k] = r2[k];
    }
    area(ics,0,v) += r1[1]*r2[2] - r2[1]*r1[2];
    area(ics,1,v) += r1[2]*r2[0] - r2[2]*r1[0];
    area(ics,2,v) += r1[0]*r2[1] - r2[0]*r1[1];
  }
  for(int k=0; k < 3; ++k)
  {
    area(ics,k,v) *= 0.5;
  }
}

  static constexpr int hex_edge_facet_table[12][4] = {
   {   20, 8, 12, 26},
   {  24,  9, 12, 26},
   {  10, 12, 26, 23},
   {  11, 25, 26, 12},
   {  13, 20, 26, 17},
   {  17, 14, 24, 26},
   {  17, 15, 23, 26},
   {  16, 17, 26, 25},
   {  19, 20, 26, 25},
   {  20, 18, 24, 26},
   {  22, 23, 26, 24},
   {  21, 25, 26, 23}};


template<int npe, int nscs>
void inline hex_scs_det(const SharedMemView<double*[3][16]>& node_coords, const SharedMemView<double*[3][16]>& area_vec,const int v)
{
  alignas(64) double coords[27][3];
  alignas(64) double scscoords[4][3];
  /*constexpr int hex_edge_facet_table[12][4] = {
   {   20, 8, 12, 26},
   {  24,  9, 12, 26},
   {  10, 12, 26, 23},
   {  11, 25, 26, 12},
   {  13, 20, 26, 17},
   {  17, 14, 24, 26},
   {  17, 15, 23, 26},
   {  16, 17, 26, 25},
   {  19, 20, 26, 25},
   {  20, 18, 24, 26},
   {  22, 23, 26, 24},
   {  21, 25, 26, 23}};
*/
  for(int i=0; i < 8; ++i)
  {
    for(int j=0; j < 3; ++j)
    {
      coords[i][j] = node_coords(i,j,v);
    }
  }

  for(int i=0; i < 3; ++i)
  {
    // Face 1
    coords[8][i] = 0.5 * (node_coords(0,i,v) + node_coords(1,i,v));
    coords[9][i] = 0.5 * (node_coords(1,i,v) + node_coords(2,i,v));
    coords[10][i] = 0.5 * (node_coords(2,i,v) + node_coords(3,i,v));
    coords[11][i] = 0.5 * (node_coords(3,i,v) + node_coords(0,i,v));
    coords[12][i] = 0.25 * (node_coords(0,i,v) + node_coords(1,i,v) + node_coords(2,i,v) + node_coords(3,i,v));

    // Face 2
    coords[13][i] = 0.5 * (node_coords(4,i,v) + node_coords(5,i,v));
    coords[14][i] = 0.5 * (node_coords(5,i,v) + node_coords(6,i,v));
    coords[15][i] = 0.5 * (node_coords(6,i,v) + node_coords(7,i,v));
    coords[16][i] = 0.5 * (node_coords(7,i,v) + node_coords(4,i,v));
    coords[17][i] = 0.25 * (node_coords(4,i,v) + node_coords(5,i,v) + node_coords(6,i,v) + node_coords(7,i,v));

    // Face 3
    coords[18][i] = 0.5 * (node_coords(1,i,v) + node_coords(5,i,v));
    coords[19][i] = 0.5 * (node_coords(0,i,v) + node_coords(4,i,v));
    coords[20][i] = 0.25 * (node_coords(0,i,v) + node_coords(1,i,v) + node_coords(4,i,v) + node_coords(5,i,v));

    // Face 4
    coords[21][i] = 0.5 * (node_coords(3,i,v) + node_coords(7,i,v));
    coords[22][i] = 0.5 * (node_coords(2,i,v) + node_coords(6,i,v));
    coords[23][i] = 0.25 * (node_coords(2,i,v) + node_coords(3,i,v) + node_coords(6,i,v) + node_coords(7,i,v));

    // Face 5
    coords[24][i] = 0.25 * (node_coords(1,i,v) + node_coords(2,i,v) + node_coords(5,i,v) + node_coords(6,i,v));

    // Face 7
    coords[25][i] = 0.25 * (node_coords(0,i,v) + node_coords(3,i,v) + node_coords(4,i,v) + node_coords(7,i,v));

    // Volume centroid
    coords[26][i] = 0.;
    for(int nd = 0; nd < 8; ++nd)
    {
      coords[26][i] += node_coords(nd,i,v);
    }
    coords[26][i] *= 0.125;
  }

  const int npf = 4;
  for(int ics=0; ics < nscs; ++ics)
  {
    for(int inode = 0; inode < npf; ++inode)
    {
      const int itrianglenode = hex_edge_facet_table[ics][inode];
      for(int d=0; d < 3; ++d)
      {
        scscoords[inode][d] = coords[itrianglenode][d];
      }
    }
    quadAreaByTriangleFacets(scscoords, area_vec,ics,v);
  }
}

template<int npe, int nint>
void inline
hex_gradient_operator(
    const SharedMemView<double*[16]>& deriv,
    const SharedMemView<double*[3][16]>& node_coords,
    SharedMemView<double*[16]>& gradop,
    SharedMemView<double*[3][16]>& detj,
    double & err,
    int & nerr)
{
/*
  double err = 0.;
  for(int ki=0; ki < nint; ++ki)
  {
    double dx_ds1 = 0.;
    double dx_ds2 = 0.;
    double dx_ds3 = 0.;
    double dy_ds1 = 0.;
    double dy_ds2 = 0.;
    double dy_ds3 = 0.;
    double dz_ds1 = 0.;
    double dz_ds2 = 0.;
    double dz_ds3 = 0.;
    for(int kn=0; kn < npe; ++kn)
    {
      dx_ds1 += deriv[ki][kn][0] * node_coords[kn][0];
      dx_ds2 += deriv[ki][kn][1] * node_coords[kn][0];
      dx_ds3 += deriv[ki][kn][2] * node_coords[kn][0];

      dy_ds1 += deriv[ki][kn][0] * node_coords[kn][1];
      dy_ds2 += deriv[ki][kn][1] * node_coords[kn][1];
      dy_ds3 += deriv[ki][kn][2] * node_coords[kn][1];

      dz_ds1 += deriv[ki][kn][0] * node_coords[kn][2];
      dz_ds2 += deriv[ki][kn][1] * node_coords[kn][2];
      dz_ds3 += deriv[ki][kn][2] * node_coords[kn][2];
    }
    double denom = dx_ds1*( dy_ds2*dz_ds3 - dz_ds2*dy_ds3 )
         + dy_ds1*( dz_ds2*dx_ds3 - dx_ds2*dz_ds3 )
         + dz_ds1*( dx_ds2*dy_ds3 - dy_ds2*dx_ds3 );
    det_j[ki] = denom;
    if( denom < std::numeric_limits<double>::min() * 1.e6)
    {
      denom = 1.;
      err = 1.;
    }
    denom = 1./denom;

    ds1_dx = denom*(dy_ds2*dz_ds3 - dz_ds2*dy_ds3)
    ds2_dx = denom*(dz_ds1*dy_ds3 - dy_ds1*dz_ds3)
    ds3_dx = denom*(dy_ds1*dz_ds2 - dz_ds1*dy_ds2)

    ds1_dy = denom*(dz_ds2*dx_ds3 - dx_ds2*dz_ds3)
    ds2_dy = denom*(dx_ds1*dz_ds3 - dz_ds1*dx_ds3)
    ds3_dy = denom*(dz_ds1*dx_ds2 - dx_ds1*dz_ds2)

    ds1_dz = denom*(dx_ds2*dy_ds3 - dy_ds2*dx_ds3)
    ds2_dz = denom*(dy_ds1*dx_ds3 - dx_ds1*dy_ds3)
    ds3_dz = denom*(dx_ds1*dy_ds2 - dy_ds1*dx_ds2)

    for(int kn=0; kn < npe; ++kn)
    {
      gradop[ki][kn][0] = deriv[ki][kn][0]*ds1_dx
        + deriv[ki][kn][1]*ds2_dx
        + deriv[ki][kn][2]*ds3_dx

      gradop[ki][kn][1] = deriv[ki][kn][0]*ds1_dy
        + deriv[ki][kn][1]*ds2_dy
        + deriv[ki][kn][2]*ds3_dy

      gradop[ki][kn][2] = deriv[ki][kn][0]*ds1_dz
        + deriv[ki][kn][1]*ds2_dz
        + deriv[ki][kn][2]
    }
  }
  if( err != 0. )
  {
    nerr = 0;
  }
*/
}
}
}

