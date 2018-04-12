#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include <sys/time.h>
#include <cuda.h>
#include <curand_kernel.h>
#include</home/joyjit/mt19937ar.h>
#include</home/joyjit/mt19937ar.c>

/* Measure done in parallel */

#define K 7  /* length of K-mer*/
#define L 128	/*Number of sites in one direction, should be even*/
#define N (L*L*L) /* The total number of sites */
#define L1 (L*L)
#define dir 3
#define T_eq 500000	/*# of MC steps for equilibration*/
#define BLOCKS 10 /*Number of blocks in which average is taken*/
#define AVE 30000000 /* # of readings in each block */
#define GAP2 10000 /*GAP after which output is written during equilibration */
#define INITIAL_FLAG 0 /*0: all empty 1: all filled */
#define BINS 101 /*Keep it odd*/
#define BINSIZE (2.0/(1.0*BINS))
#define nthreads 8
#define nsize (K*K)
#define nturns (N/nsize)
#define LBS (L/K)

const int threadsPerBlock = 256;
const int  blocksPerGrid  = (L1+threadsPerBlock-1)/threadsPerBlock;

int taskid,noh,nov,nod;
float periodic[K],acceptance[L+1],mu;
double count[BINS],meanrho[BLOCKS],meanabs[BLOCKS],rho2[BLOCKS],fluc[BLOCKS];
double meanm1[BLOCKS],meanm2[BLOCKS],meanm4[BLOCKS];
double Q_1[BLOCKS],Q_2[BLOCKS],Q_4[BLOCKS],Rho_2[BLOCKS],Rho[BLOCKS],modQ[BLOCKS],kappa[BLOCKS],chi[BLOCKS];
double mass[BINS],factor[BINS],bincount[BINS];
char outfile1[100],outfile2[100],outfile3[100];
char readfile1[100],readfile2[100];
int *lat;
int *sum1;
int *sum2;
int *sum3;

void take_input()
{
	long seedval;
	seedval=919737;
	//scanf("%ld",&seedval);
	init_genrand(seedval);
}

__constant__ float dev_acc[L+1];
__constant__ float dev_per[K];

__device__ int rn(int i)
{
	int x,y,z,tmp;
	z=i/L1;
	y=(i%L1)/L;
	x=(i%L1)%L;

	tmp = (x == L - 1 ? 0 : x + 1) + L * y + L * L * z;
	
	return tmp;
}

__device__ int ln(int i)
{
	int x,y,z,tmp;
	z=i/L1;
	y=(i%L1)/L;
	x=(i%L1)%L;

	tmp = (x == 0 ? L-1 : x-1) + L * y + L * L * z;
	
	return tmp;
}

__device__ int tn(int i)
{
	
	int x,y,z,tmp;
	z=i/L1;
	y=(i%L1)/L;
	x=(i%L1)%L;
	
	tmp= x + L * (y == L - 1 ? 0 : y + 1) + L * L * z;
	
	return tmp;
}

__device__ int bn(int i)
{

	int x,y,z,tmp;
	z=i/L1;
	y=(i%L1)/L;
	x=(i%L1)%L;
	
	tmp = x + L * (y == 0 ? L-1 : y-1) + L * L * z;

	return tmp;
}

__device__ int rtn(int i)
{

	int x,y,z,tmp;
	z=i/L1;
	y=(i%L1)/L;
	x=(i%L1)%L;
	
	tmp = x + L * y + L * L * (z == L - 1 ? 0 : z + 1);

	return tmp;
}

__device__ int lbn(int i)
{

	int x,y,z,tmp;
	z=i/L1;
	y=(i%L1)/L;
	x=(i%L1)%L;

	tmp = x + L * y + L * L * (z == 0 ? L-1 : z-1);

	return tmp;
}



void initialize()
{
	/* initializes nbr list and output file names */
	int i,j;
	double tmp,tmp1;

	if(INITIAL_FLAG==0)
		sprintf(outfile1,"emptyK%dL%dM%5.4lf_omp02",K,L,mu);
	if(INITIAL_FLAG==1)
		sprintf(outfile1,"filledK%dL%dM%5.4lf_omp02",K,L,mu);
	if(INITIAL_FLAG==2)
                sprintf(outfile1,"halfK%dL%dM%5.4lf_omp02",K,L,mu);
	/*initializing bin parameters*/

	for(j=0;j<BINS;j++)
	{
		bincount[j]=0;
		mass[j]=0;
	}
	for(j=0;j<=N;j=j+K)
	{
		tmp=(1.0*j)/(1.0*N);
		i=floor((tmp)/BINSIZE);
		/*if(j==-N)
			i=0;*/
		if(j==N)
			i=BINS-1;
		bincount[i]++;
		mass[i]=mass[i]+tmp;
	}
	tmp1=(K*1.0)/(1.0*N);
	for(j=0;j<BINS;j++)
	{
		mass[j]=mass[j]/bincount[j];
		factor[j]=tmp1*bincount[j];
	}
}

void lat_init()
{
	/* Initializes quantities that have to initialized for every value
	 * of probability p */
	int i;
	double x;
	FILE *fp;
	for(i=0;i<BLOCKS;i++)
	{
		meanrho[i]=0;meanabs[i]=0;meanm1[i]=0; meanm2[i]=0; meanm4[i]=0,rho2[i]=0,fluc[i]=0;
		Rho[i]=0;Rho_2[i]=0;modQ[i]=0;Q_1[i]=0;Q_2[i]=0;Q_4[i]=0; kappa[i]=0; chi[i]=0;
	}
	/*if((L % K !=0)|| (L % 2) !=0)
	{
		printf("ERROR IN DIVISIBILITY\n");
		exit(0);
	}*/
	
	if(INITIAL_FLAG==0)
	{	
		sprintf(outfile2,"emptyK%dL%dM%5.4lfTPB%domp02_t",K,L,mu,threadsPerBlock);
		sprintf(outfile3,"emptyK%dL%dM%5.4lfomp02_p",K,L,mu,threadsPerBlock);
	}
	if(INITIAL_FLAG==1)
	{
		sprintf(outfile2,"filledK%dL%dM%5.4lfTPB%domp02_t",K,L,mu,threadsPerBlock);
		sprintf(outfile3,"filledK%dL%dM%5.4lfTPB%domp02_p",K,L,mu,threadsPerBlock);
	}
	if(INITIAL_FLAG==2)
        {
                sprintf(outfile2,"halfK%dL%dM%5.4lfTPB%domp02_t",K,L,mu,threadsPerBlock);
                sprintf(outfile3,"halfK%dL%dM%5.4lfTPB%domp02_p",K,L,mu,threadsPerBlock);
        }
	fp=fopen(outfile2,"w");
	fprintf(fp,"#t rho abs(m) m\n");
	fclose(fp);

	sprintf(readfile1,"acceptprobK%dL%dM%5.4lf",K,L,mu);
	fp=fopen(readfile1,"r");
	if(fp==NULL)
	{
		printf("The FILE [%s] DOES NOT EXIST\n",readfile1);
		exit(0);
	}
	while(fscanf(fp,"%d%lf",&i,&x)!=EOF)
		acceptance[i]=x;
	fclose(fp);
	if(i!=L)
	{
		printf("Error in FILE [%s]\n",readfile1);
		exit(0);
	}

	sprintf(readfile2,"periodicprobK%dL%dM%5.4lf",K,L,mu);
	fp=fopen(readfile2,"r");
	if(fp==NULL)
	{
		printf("The FILE [%s] DOES NOT EXIST\n",readfile2);
		exit(0);
	}
	while(fscanf(fp,"%d%lf",&i,&x)!=EOF)
		periodic[i]=x;
	fclose(fp);
	if(i!=K-1)
	{
		printf("Error in FILE [%s]\n",readfile2);
		exit(0);
	}

}

__device__ void deposit_hor(int i, int *lat)
{
	/*puts a horizontal kmer with head at i*/

	int j;
	
	lat[i]=-1;
	for(j=1;j<K;j++)
	{
		i=rn(i);
		lat[i]=1;
	}
}

__device__ void deposit_ver(int i, int *lat)
{
	/*puts a vertical kmer with head at i*/

	int j;
	
	lat[i]=-2;
	for(j=1;j<K;j++)
	{
		i=tn(i);
		lat[i]=2;
	}
}

__device__ void deposit_diag(int i, int *lat)
{
	/*puts a vertical kmer with head at i*/

	int j;
	
	lat[i]=-3;
	for(j=1;j<K;j++)
	{
		i=rtn(i);
		lat[i]=3;
	}
}

__device__ int find_starthfinalh(int id, int row, int *starth, int *finalh, int *lat, curandState* randstate)
{
	int i,k;

	*starth=row; *finalh=row;
	if(lat[*starth]!=0)
	{
		while(lat[*starth]!=0)
		{
			*starth=rn(*starth);
			if(*starth==row)
				return 0;
		}
		while(lat[*finalh]!=0)
			*finalh=ln(*finalh);
		return 1;
	}

	while(lat[ln(*starth)]==0)
	{
		*starth=ln(*starth);
		if(*starth==row)
		{
			*finalh=ln(*starth);
			for(i=0;i<K-1;i++)
			{
				if(curand_uniform((randstate+id)) < dev_per[i])
				{			
					lat[*starth]=-1;
					*starth=rn(*starth);
					for(k=2;k<=K;k++)
					{
						lat[*starth]=1;
						*starth=rn(*starth);
					}
					break;
				}
				else
				{
					*finalh=*starth;
					*starth=rn(*starth);
				}
			}
			return 1;
		}
	}
	*finalh=ln(*starth);
	while(lat[*finalh]!=0)
		*finalh=ln(*finalh);
	return 1;
}

__device__ void fill_row(int id, int row, int *lat, curandState* randstate)
{
	/*fills row 'row' with horizontal kmers*/

	int i,len,starth,finalh,endh;

	if(find_starthfinalh(id, row, &starth, &finalh, lat, randstate)==0)
		return;
	do
	{
		endh=starth;len=1;
		while(lat[rn(endh)]==0)
		{			
			endh=rn(endh);
			len++;
			if(endh==finalh)		
				break;
		}
		while(len>=K)
		{
			if(curand_uniform((randstate+id)) < dev_acc[len])
			{
				deposit_hor(starth,lat);
				for(i=0;i<K;i++)
					starth=rn(starth);
				len=len-K;
			}
			else
			{
				starth=rn(starth);
				len--;
			}
		}
		if(endh==finalh)
			return;
		starth=rn(endh);
		while(lat[starth]!=0)
			starth=rn(starth);
	}
	while(endh!=finalh);
}

__device__ int find_startvfinalv(int id, int col, int *startv, int *finalv, int *lat, curandState* randstate)
{
	/*in row 'row' finds out startv and finalv*/

	int i,k;

	*startv=col; *finalv=col;
	if(lat[*startv]!=0)
	{
		while(lat[*startv]!=0)
		{
			*startv=tn(*startv);
			if(*startv==col)
				return 0;
		}
		while(lat[*finalv]!=0)
			*finalv=bn(*finalv);
		return 1;
	}

	while(lat[bn(*startv)]==0)
	{
		*startv=bn(*startv);
		if(*startv==col)
		{
			*finalv=bn(*startv);
			for(i=0;i<K-1;i++)
			{
				if(curand_uniform((randstate+id)) < dev_per[i])
				{				
					lat[*startv]=-2;
					*startv=tn(*startv);
					for(k=2;k<=K;k++)
					{
						lat[*startv]=2;
						*startv=tn(*startv);
					}
					break;
				}
				else
				{
					*finalv=*startv;
					*startv=tn(*startv);
				}
			}
			return 1;
		}
	}
	*finalv=bn(*startv);
	while(lat[*finalv]!=0)
		*finalv=bn(*finalv);
	return 1;
}

__device__ void fill_col(int id, int col, int *lat, curandState* randstate)
{
	/*fills col 'col' with horizontal kmers*/

	int i,len,startv,finalv,endv;

	if(find_startvfinalv(id, col, &startv, &finalv, lat, randstate)==0)
		return;
	do
	{
		endv=startv;len=1;
		while(lat[tn(endv)]==0)
		{
			endv=tn(endv);
			len++;
			if(endv==finalv)
				break;
		}
		while(len>=K)
		{
			if(curand_uniform((randstate+id)) < dev_acc[len])
			{
				deposit_ver(startv,lat);
				for(i=0;i<K;i++)
					startv=tn(startv);
				len=len-K;
			}
			else
			{
				startv=tn(startv);
				len--;
			}
		}
		if(endv==finalv)
			return;
		startv=tn(endv);
		while(lat[startv]!=0)
			startv=tn(startv);
	}
	while(endv!=finalv);
}

__device__ int find_startdfinald(int id, int diag, int *startd, int *finald, int *lat, curandState* randstate)
{
	/*in row 'row' finds out starth and finalh*/

	int i,k;

	*startd=diag; *finald=diag;
	if(lat[*startd]!=0)
	{
		while(lat[*startd]!=0)
		{
			*startd=rtn(*startd);
			if(*startd==diag)
				return 0;
		}
		while(lat[*finald]!=0)
			*finald=lbn(*finald);
		return 1;
	}

	while(lat[lbn(*startd)]==0)
	{
		*startd=lbn(*startd);
		if(*startd==diag)
		{
			*finald=lbn(*startd);
			for(i=0;i<K-1;i++)
			{
				if(curand_uniform((randstate+id)) < dev_per[i])
				{
					lat[*startd]=-3;
					*startd=rtn(*startd);
					for(k=2;k<=K;k++)
					{
						lat[*startd]=3;
						*startd=rtn(*startd);
					}
					break;
				}
				else
				{
					*finald=*startd;
					*startd=rtn(*startd);
				}
			}
			return 1;
		}
	}
	*finald=lbn(*startd);
	while(lat[*finald]!=0)
		*finald=lbn(*finald);
	return 1;
}

__device__ void fill_diag(int id, int diag, int *lat, curandState* randstate)
{
	/*fills row 'row' with horizontal kmers*/

	int i,len,startd,finald,endd;

	if(find_startdfinald(id, diag, &startd, &finald, lat, randstate)==0)
		return;
	do
	{
		endd=startd;len=1;
		while(lat[rtn(endd)]==0)
		{
			endd=rtn(endd);
			len++;
			if(endd==finald)
				break;
		}
		while(len>=K)
		{
			if(curand_uniform((randstate+id)) < dev_acc[len])
			{
				deposit_diag(startd,lat);
				for(i=0;i<K;i++)
					startd=rtn(startd);
				len=len-K;
			}
			else
			{
				startd=rtn(startd);
				len--;
			}
		}
		if(endd==finald)
			return;
		startd=rtn(endd);
		while(lat[startd]!=0)
			startd=rtn(startd);
	}
	while(endd!=finald);
}

__global__ void setup_kernel ( curandState* randstate, unsigned long int seed )
{
    //int i = threadIdx.x;
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init ( seed, i, 0, (randstate+i) );
} 

__global__ void row(int *lat, curandState* randstate)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int i,row,site;
	row=tid*L;
	site=row;
	if (tid < (L*L)) 
	{
		/*remove rod */
		for(i=0;i<L;i++)
		{
			if((lat[site]==1)||(lat[site]==-1))
				lat[site]=0;
			site=rn(site);
		}

		fill_row(tid, row, lat, randstate);
	}
}

__global__ void col( int *lat, curandState* randstate)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int i,col,site;
	col=(tid % L)+(tid / L)*L*L;
	site=col;
	if (tid < (L*L)) 
	{
		/*remove rod */
		for(i=0;i<L;i++)
		{	
			if((lat[site]==2)||(lat[site]==-2))
					lat[site]=0;
				site=tn(site);
		}

		fill_col(tid, col, lat, randstate);
	}
}

__global__ void diag( int *lat, curandState* randstate)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int i,diag,site;
	diag=tid;
	site=diag;
	if (tid < (L*L)) 
	{
		/*remove rod */
		for(i=0;i<L;i++)
		{	
			if((lat[site]==3)||(lat[site]==-3))
				lat[site]=0;
			site=rtn(site);
		}

		fill_diag(tid, diag, lat, randstate);
	}
}

__global__ void calculate(int *lat, int *sum1, int *sum2, int *sum3)
{
	int i,tmph,tmpv,tmpd;
	int tid=threadIdx.x + blockIdx.x * blockDim.x;
	int cacheindex=threadIdx.x;
	int row=tid;
	tmph=0; tmpv=0; tmpd=0;
	__shared__ int cache1[threadsPerBlock];
	__shared__ int cache2[threadsPerBlock];
	__shared__ int cache3[threadsPerBlock];
	
	for(i=row*L;i<(row+1)*L;i++)
	{
		if(lat[i]<0)
		{
			if(lat[i] == -1)
				tmph++;
			else if(lat[i] == -2)
				tmpv++;		
			else 
				tmpd++;		
		}
	}

	cache1[cacheindex]=tmph; cache2[cacheindex]=tmpv; cache3[cacheindex]=tmpd; 
	__syncthreads();

	int bi = blockDim.x/2;
	while(bi != 0)
	{
		if(cacheindex < bi)
		{
			cache1[cacheindex]=cache1[cacheindex]+cache1[cacheindex+bi];
			cache2[cacheindex]=cache2[cacheindex]+cache2[cacheindex+bi];
			cache3[cacheindex]=cache3[cacheindex]+cache3[cacheindex+bi];
		}
		__syncthreads();
		bi=bi/2;
	}

	/*bi = blockDim.x/2;
	while(bi != 0)
	{
		if(cacheindex < bi)
			cache2[cacheindex]=cache2[cacheindex]+cache2[cacheindex+bi];
		__syncthreads();
		bi=bi/2;
	}*/
	
	if(cacheindex == 0)
	{
		sum1[blockIdx.x]= cache1[0];
		sum2[blockIdx.x]= cache2[0];
		sum3[blockIdx.x]= cache3[0];
	}
}

void output2(int t)
{
	int i;
	double tmp;
	FILE *fp;
	noh=0; nov=0; nod=0;
	for(i=0;i<blocksPerGrid;i++)
	{
		noh=noh+sum1[i];
		nov=nov+sum2[i];
		nod=nod+sum3[i];
	}
	noh=K*noh; nov=K*nov; nod=nod*K;
	tmp=1.0*(sqrt(pow((noh-1.0*nov/2.0-1.0*nod/2.0),2.0)+3.0*(pow((nov-nod),2.0))/4.0))/(1.0*N);
	fp=fopen(outfile2,"a");
	fprintf(fp,"%d\t%e\t%e\t%e\t%e\t%e\n",t,1.0*(noh+nov+nod)/(1.0*N),1.0*noh/N,1.0*nov/N,1.0*nod/N,tmp);
	fclose(fp);
}

/*int hor_compatibility(int inp)
{
	int i,tmp;
	tmp=inp;
	
	for(i=0;i<K;i++)
	{
		if(lat[tmp]!=0)
			return 0;
		tmp=rn(tmp);			
	}
	return 1;
}

int ver_compatibility(int inp)
{
	int i,tmp;
	tmp=inp;
	
	for(i=0;i<K;i++)
	{
		if(lat[tmp]!=0)
			return 0;
		tmp=tn(tmp);	
	}
	return 1;
}

int diag_compatibility(int inp)
{
	int i,tmp;
	tmp=inp;
	
	for(i=0;i<K;i++)
	{
		if(lat[tmp]!=0)
			return 0;
		tmp=rtn(tmp);	
	}
	return 1;
}*/

void lat_initialization()
{
	int i,j,k;
	//double rand;

	if(INITIAL_FLAG == 0)
	{
		for(i=0;i<N;i++)
			lat[i]=0;
	}
	/*else if(INITIAL_FLAG == 1) 
	{
		for(i=0;i<(2*N);i++)
		{
			st=floor(genrand_real3()*N);
			if(lat[st] == 0)
			{
				rand=genrand_real3();
				if(rand < 0.33)
				{
					if(hor_compatibility(st)==1)	
						deposit_hor(st,lat);
				}				
				else if(rand < 0.66)
				{
					if(ver_compatibility(st)==1)
						deposit_ver(st,lat);
				}
				else 
				{
					if(diag_compatibility(st)==1)	
						deposit_diag(st,lat);
				}
				
			}
		}
	}*/
	else
	{
		for(i=0;i<N/2;i++)
			lat[i]=1;
		for(i=0;i<(N/2/K);i++)
			lat[i*K]=-1;
		for(i=N/2;i<N;i++)
                        lat[i]=2;
		for(i=L/2;i<L;i++)
		{
			for(j=0;j<(L/K);j++)
			{
				for(k=0;k<L;k++)
					lat[i*L*L+j*L*K+k]=-2;
			}
		}
	}
}

int main (int argc, char *argv[])
{
	int ms; 
	int *dev_lat, *dev_sum1, *dev_sum2, *dev_sum3;

	/* allocating memory on CPU */
	lat = (int *) malloc (N * sizeof(int));
	sum1= (int *) malloc (blocksPerGrid * sizeof(int));
	sum2= (int *) malloc (blocksPerGrid * sizeof(int));
	sum3= (int *) malloc (blocksPerGrid * sizeof(int));

	/* allocating memory on GPU */
	cudaMalloc((void **) &dev_lat, N * sizeof(int));
	cudaMalloc((void **) &dev_sum1, blocksPerGrid * sizeof(int));
	cudaMalloc((void **) &dev_sum2, blocksPerGrid * sizeof(int));
	cudaMalloc((void **) &dev_sum3, blocksPerGrid * sizeof(int));

	//neighbour();
	mu=1.75;
	//scanf("%lf",&mu);
	take_input();
	initialize();
	lat_init();
	lat_initialization();

	/* copy the arrays from CPU to the GPU */
	cudaMemcpy( dev_lat, lat, N*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol( dev_acc, acceptance, sizeof(acceptance));
	cudaMemcpyToSymbol( dev_per, periodic, sizeof(periodic));

	curandState *randstate;
	cudaMalloc ( &randstate, L1*sizeof(*randstate ) );
	// setup seeds
	setup_kernel <<<blocksPerGrid,threadsPerBlock>>> (randstate, time(NULL) );

	// evolve	
	for(ms=0;ms<T_eq;ms++)
	{	
		row<<<blocksPerGrid,threadsPerBlock>>>( dev_lat, randstate );

		col<<<blocksPerGrid,threadsPerBlock>>>( dev_lat, randstate );

		diag<<<blocksPerGrid,threadsPerBlock>>>( dev_lat, randstate );

		//calculate<<<blocksPerGrid,threadsPerBlock>>>(dev_lat, dev_sum1, dev_sum2);

		if(ms % GAP2 == 0)
		{	
			calculate<<<blocksPerGrid,threadsPerBlock>>>(dev_lat, dev_sum1, dev_sum2, dev_sum3);
			cudaMemcpy( sum1, dev_sum1, blocksPerGrid*sizeof(int),cudaMemcpyDeviceToHost);
			cudaMemcpy( sum2, dev_sum2, blocksPerGrid*sizeof(int),cudaMemcpyDeviceToHost);
			cudaMemcpy( sum3, dev_sum3, blocksPerGrid*sizeof(int),cudaMemcpyDeviceToHost);
			output2(ms);
		}
	}

	
	// free the memory allocated on the CPU
	free(lat); free(sum1); free(sum2); free(sum3);

	// free the memory allocated on the GPU
	cudaFree( dev_lat); cudaFree( dev_sum1); cudaFree( dev_sum2); cudaFree( dev_sum3); 	
	cudaFree(randstate);
	return 0;

}
