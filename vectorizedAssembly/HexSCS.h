namespace sierra {
namespace nalu {

  static constexpr int triangularFacetTable[4][3] = {
    {4, 0, 1},
    {4, 1, 2},
    {4, 2, 3},
    {4, 3, 0}};
#pragma omp declare simd
  __attribute__((vector(processor(core_2nd_gen_avx))))
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
  constexpr int ntriangles = 4;
  alignas(64) double r1[3];
  for(int itriangle = 0; itriangle < ntriangles; ++itriangle)
  {
    const int iq = triangularFacetTable[itriangle][2];
    for(int k=0; k < 3; ++k)
    {
      r1[k] = r2[k];
      r2[k] = areacoords[iq][k] - xmid[k];
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


#pragma omp declare simd
template<int npe, int nscs>
  __attribute__((vector(processor(core_2nd_gen_avx))))
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

  constexpr int npf = 4;
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


#pragma omp declare simd
template<int npe, int nint>
  __attribute__((vector(processor(core_2nd_gen_avx))))
void inline
hex_derivative(
    SharedMemView<double*[8][3][16]>& deriv,
    int v
    )
{
      double half, one4th;
      half = 1.0/2.0;
      one4th = 1.0/4.0;
      constexpr double intgLoc_[12][3] = {
      { 0.00, -0.25,  -0.25},
       {0.25,  0.00,-0.25},
       {0.00,  0.25,-0.25},
      {-0.25,  0.00,-0.25},
       {0.00, -0.25, 0.25},
       {0.25,  0.00, 0.25},
       {0.00,  0.25, 0.25},
      {-0.25,  0.00, 0.25},
      {-0.25, -0.25, 0.00},
       {0.25, -0.25, 0.00},
       {0.25,  0.25, 0.00},
      {-0.25,  0.25, 0.00} };

      for (int j = 0; j < nint; ++j)
      {
         double s1 = intgLoc_[j][0];
         double s2 = intgLoc_[j][1];
         double s3 = intgLoc_[j][2];
         double s1s2 = s1*s2;
         double s2s3 = s2*s3;
         double s1s3 = s1*s3;

         // shape function derivative in the s1 direction -
         deriv(j,0,0,v) = half*( s3 + s2 ) - s2s3 - one4th;
         deriv(j,1,0,v) = half*(-s3 - s2 ) + s2s3 + one4th;
         deriv(j,2,0,v) = half*(-s3 + s2 ) - s2s3 + one4th;
         deriv(j,3,0,v) = half*(+s3 - s2 ) + s2s3 - one4th;
         deriv(j,4,0,v) = half*(-s3 + s2 ) + s2s3 - one4th;
         deriv(j,5,0,v) = half*(+s3 - s2 ) - s2s3 + one4th;
         deriv(j,6,0,v) = half*(+s3 + s2 ) + s2s3 + one4th;
         deriv(j,7,0,v) = half*(-s3 - s2 ) - s2s3 - one4th;
         //
         // shape function derivative in the s2 direction -
         deriv(j,0,1,v) = half*( s3 + s1 ) - s1s3 - one4th;
         deriv(j,1,1,v) = half*( s3 - s1 ) + s1s3 - one4th;
         deriv(j,2,1,v) = half*(-s3 + s1 ) - s1s3 + one4th;
         deriv(j,3,1,v) = half*(-s3 - s1 ) + s1s3 + one4th;
         deriv(j,4,1,v) = half*(-s3 + s1 ) + s1s3 - one4th;
         deriv(j,5,1,v) = half*(-s3 - s1 ) - s1s3 - one4th;
         deriv(j,6,1,v) = half*( s3 + s1 ) + s1s3 + one4th;
         deriv(j,7,1,v) = half*( s3 - s1 ) - s1s3 + one4th;

         // shape function derivative in the s3 direction -
         deriv(j,0,2,v) = half*( s2 + s1 ) - s1s2 - one4th;
         deriv(j,1,2,v) = half*( s2 - s1 ) + s1s2 - one4th;
         deriv(j,2,2,v) = half*(-s2 - s1 ) - s1s2 - one4th;
         deriv(j,3,2,v) = half*(-s2 + s1 ) + s1s2 - one4th;
         deriv(j,4,2,v) = half*(-s2 - s1 ) + s1s2 + one4th;
         deriv(j,5,2,v) = half*(-s2 + s1 ) - s1s2 + one4th;
         deriv(j,6,2,v) = half*( s2 + s1 ) + s1s2 + one4th;
         deriv(j,7,2,v) = half*( s2 - s1 ) - s1s2 + one4th;
      }
}

#pragma omp declare simd
template<int npe, int nint>
  __attribute__((vector(processor(core_2nd_gen_avx))))
void inline
hex_gradient_operator(
    const SharedMemView<double*[8][3][16]>& deriv,
    const SharedMemView<double*[3][16]>& node_coords,
    SharedMemView<double*[8][3][16]>& gradop,
    SharedMemView<double*[16]>& detj,
    double & err,
    int & nerr,
    int v)
{
  err = 0.;
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
      dx_ds1 += deriv(ki, kn, 0, v) * node_coords(kn, 0, v);
      dx_ds2 += deriv(ki, kn, 1, v) * node_coords(kn, 0, v);
      dx_ds3 += deriv(ki, kn, 2, v) * node_coords(kn, 0, v);

      dy_ds1 += deriv(ki, kn, 0, v) * node_coords(kn, 1, v);
      dy_ds2 += deriv(ki, kn, 1, v) * node_coords(kn, 1, v);
      dy_ds3 += deriv(ki, kn, 2, v) * node_coords(kn, 1, v);

      dz_ds1 += deriv(ki, kn, 0, v) * node_coords(kn, 2, v);
      dz_ds2 += deriv(ki, kn, 1, v) * node_coords(kn, 2, v);
      dz_ds3 += deriv(ki, kn, 2, v) * node_coords(kn, 2, v);
    }
    double denom = dx_ds1*( dy_ds2*dz_ds3 - dz_ds2*dy_ds3 )
         + dy_ds1*( dz_ds2*dx_ds3 - dx_ds2*dz_ds3 )
         + dz_ds1*( dx_ds2*dy_ds3 - dy_ds2*dx_ds3 );
    detj(ki, v) = denom;
    if( denom < std::numeric_limits<double>::min() * 1.e6)
    {
      denom = 1.;
      err = 1.;
    }
    denom = 1./denom;

    double ds1_dx = denom*(dy_ds2*dz_ds3 - dz_ds2*dy_ds3);
    double ds2_dx = denom*(dz_ds1*dy_ds3 - dy_ds1*dz_ds3);
    double ds3_dx = denom*(dy_ds1*dz_ds2 - dz_ds1*dy_ds2);

    double ds1_dy = denom*(dz_ds2*dx_ds3 - dx_ds2*dz_ds3);
    double ds2_dy = denom*(dx_ds1*dz_ds3 - dz_ds1*dx_ds3);
    double ds3_dy = denom*(dz_ds1*dx_ds2 - dx_ds1*dz_ds2);

    double ds1_dz = denom*(dx_ds2*dy_ds3 - dy_ds2*dx_ds3);
    double ds2_dz = denom*(dy_ds1*dx_ds3 - dx_ds1*dy_ds3);
    double ds3_dz = denom*(dx_ds1*dy_ds2 - dy_ds1*dx_ds2);

    for(int kn=0; kn < npe; ++kn)
    {
      gradop(ki, kn, 0, v) =
          deriv(ki, kn, 0, v)*ds1_dx
        + deriv(ki, kn, 1, v)*ds2_dx
        + deriv(ki, kn, 2, v)*ds3_dx;

      gradop(ki, kn, 1, v) =
          deriv(ki, kn, 0, v)*ds1_dy
        + deriv(ki, kn, 1, v)*ds2_dy
        + deriv(ki, kn, 2, v)*ds3_dy;

      gradop(ki, kn, 2, v) =
          deriv(ki, kn, 0, v)*ds1_dz
        + deriv(ki, kn, 1, v)*ds2_dz
        + deriv(ki, kn, 2, v)*ds3_dz;
    }
  }
  if( err != 0. )
  {
    nerr = 0;
  }
}
}
}

