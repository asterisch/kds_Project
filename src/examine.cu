#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define LSIZE 31
#define MIN_LIM 12.0
#define MAX_LIM 30.0
#define BLOCKSIZE 32
void check_input(int argc,char* argv[]);
__global__ void examine(float *d_coordinates,int *d_coords_within,int d_lines);
long calc_lines(char *filename);

int main(int argc,char * argv[])
{
  check_input(argc,argv);
  char *filename=argv[3];
  int coll = atoi(argv[1]);
  int threads=atoi(argv[4]);
  long loop_count;
  loop_count =calc_lines(filename);						// Count the lines of input file
  FILE *input=fopen(filename,"r");
  if(coll != -1)													// Handle max_collisions argument
  {
    if(coll>loop_count)
    {
      printf("[!] Warning: Specified collisions to be tested exceed the ones in input file\n");
      printf("[!] Setting the number of collisions to the maximum (taken from input file)\n");
    }
    else
    {
      loop_count = coll;
    }
  }
  if (threads!=-1)
  {

  }
  else
  {
    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop,0);
    threads=prop.maxThreadsPerMultiProcessor;
  }

  dim3 blockSize(BLOCKSIZE);
  dim3 gridSize(threads/BLOCKSIZE);
  float *h_coordinates=(float * )malloc(3*threads*sizeof(float));
  float *d_coordinates;
  int *d_coords_within,*h_coords_within=(int*)malloc(sizeof(int));
  *h_coords_within=0;
  cudaMalloc(&d_coordinates,3*threads*sizeof(float));
  cudaMalloc(&d_coords_within,sizeof(int));
  cudaMemcpy(d_coords_within,h_coords_within,sizeof(int),cudaMemcpyHostToDevice);
  int i,j=0;
  while(j<loop_count)
  {
    if (j+threads>loop_count)
    {
      threads=loop_count-j;
      cudaFree(d_coordinates);
      cudaMalloc(&d_coordinates,3*threads*sizeof(float));

    }
    for(i=0;i<threads;i++)
    {
      fscanf(input,"%f %f %f",&h_coordinates[i*3],&h_coordinates[i*3+1],&h_coordinates[i*3+2]);
    }
    cudaMemcpy(d_coordinates,h_coordinates,3*threads*sizeof(float),cudaMemcpyHostToDevice);
    examine<<<gridSize,blockSize>>>(d_coordinates,d_coords_within,3*threads);
    cudaMemcpy(h_coords_within,d_coords_within,sizeof(int),cudaMemcpyDeviceToHost);

    j+=threads;
  }
  cudaMemcpy(h_coords_within,d_coords_within,sizeof(int),cudaMemcpyDeviceToHost);
  cudaFree(d_coordinates);
  cudaFree(d_coords_within);
  fclose(input);
  printf("coordinates within limits are %d\n",*h_coords_within );
  free(h_coordinates);
  free(h_coords_within);
  return 0;
}
__global__ void examine(float *d_coordinates,int *d_coords_within,int d_lines)
{
    int index=blockIdx.x*blockDim.x+3*threadIdx.x;
    if(index>=d_lines) return;
    if(d_coordinates[index] >= MIN_LIM && d_coordinates[index] <= MAX_LIM && d_coordinates[index+1] >= MIN_LIM && d_coordinates[index+1] <= MAX_LIM && d_coordinates[index+2] >= MIN_LIM && d_coordinates[index+2] <= MAX_LIM)
    {						                           
      		atomicAdd((unsigned int*)d_coords_within,1);						// If the current coordinate is within the accepted limits,
    }


}
void check_input(int argc,char *argv[]) 																		// Handle number of arguments errors and show usage
{
	if (argc<5 || argc>5)
	{
		printf("[-] Usage: ./examine [max_collisions] [max_exec_time] [input_file] [Threads] \nUse \"-1\": for no boundies \n");
		if (argc==2) if (!strcmp(argv[1],"--help")) printf("max_collisions: Maximum number of collisions\nmax_exec_time: Maximum execution time\ninput_file: Filename to examine\nThreads: Number of threads to use\n" );
		exit(2);
	}
}
long calc_lines(char *filename) 																						// Calculates the lines of input file
{
	FILE *file=fopen(filename,"r");
	fseek(file,0L,SEEK_END);															//set file position indicator right to the end-of-file
	long lines=ftell(file);																//store the number of bytes since the beginning of the file
	fseek(file,0L,SEEK_SET);
	fclose(file);
	return lines/LSIZE;																			//return lines count of the file
}
