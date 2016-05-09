#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define LSIZE 31
#define MIN_LIM 12.0
#define MAX_LIM 30.0

void check_input(int argc,char* argv[]);
__global__ void examine(float *d_coordinates,int *d_coords_within,int d_lines);
long calc_lines(char *filename);

int main(int argc,char * argv[])
{
  check_input(argc,argv);
  char *filename=argv[3];
  int coll = atoi(argv[1]);
  int exec_time=atoi(argv[2]);
  int threads=atoi(argv[4]);
  int BLOCKSIZE = atoi(argv[5]);
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
  if (BLOCKSIZE==-1)
  {
      BLOCKSIZE=256;
  }
  if (threads!=-1)
  {
    if (threads%BLOCKSIZE!=0)
    {
      threads=(threads/BLOCKSIZE)*BLOCKSIZE;
    }
  }
  else
  {
    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop,0);
    threads=prop.maxThreadsPerMultiProcessor;
  }
  printf("[+] Using %d GPU-Threads with BlockSize %d\n",threads,BLOCKSIZE );
  cudaEvent_t start,stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

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
  float time_elapsed = 0;
  printf("[+] Working...\n" );
  cudaEventRecord(start);           // Starting time reference
  while(j<loop_count && (exec_time==-1?1:time_elapsed<exec_time))
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
    cudaEventRecord(stop);                               // Stop time reference
    cudaEventSynchronize(stop);                         // Block CPU until "stop" event is recorded
    cudaEventElapsedTime(&time_elapsed, start, stop);  // Calculate the time elapsed in milliseconds
    time_elapsed=time_elapsed/1000;                    // Convert milliseconds to seconds
    j+=threads;
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaMemcpy(h_coords_within,d_coords_within,sizeof(int),cudaMemcpyDeviceToHost);
  //Printing results
  loop_count=j;
  printf("[+] Main part of the program was being executed for :: %.3f :: sec)\n", time_elapsed);
  printf("[+] %d coordinates have been read\n[+] %d cooordinates were inside the area of interest\n[+] %d coordinates read per second\n", loop_count, *h_coords_within, (time_elapsed<=0?loop_count:loop_count/(int)time_elapsed));

  cudaFree(d_coordinates);
  cudaFree(d_coords_within);
  fclose(input);
  free(h_coordinates);
  free(h_coords_within);

  return 0;
}
__global__ void examine(float *d_coordinates,int *d_coords_within,int d_lines)
{
    int index=blockIdx.x*3*blockDim.x+3*threadIdx.x;
    float coord1=d_coordinates[index],coord2=d_coordinates[index+1],coord3=d_coordinates[index+2];
    if(index>=d_lines) return;
    if(coord1 >= MIN_LIM && coord1 <= MAX_LIM && coord2 >= MIN_LIM && coord2 <= MAX_LIM && coord3 >= MIN_LIM && coord3 <= MAX_LIM)
    {
              atomicAdd((unsigned int*)d_coords_within,1); // So as threads do not mess up the values     								// If the current coordinate is within the accepted limits,
    }


}
void check_input(int argc,char *argv[]) 																		// Handle number of arguments errors and show usage
{
	if (argc<6 || argc>6)
	{
		printf("[-] Usage: ./examine [max_collisions] [max_exec_time] [input_file] [Threads] [1D_blockSize]\nUse \"-1\": for no boundies \n");
		if (argc==2) if (!strcmp(argv[1],"--help"))
    {
      printf("max_collisions: Maximum number of collisions\nmax_exec_time: Maximum execution time\ninput_file: Filename to examine\nThreads: Number of gpu-threads to use\n1D_blocksize: gpu-blocksize to use" );
      printf("\t ======Usefull info!======\n");
      printf("1) 1D_blockSize must be a multiple of 32. (or whatever warp_size is supported by your GPU)\n2) Threads must be a multiple of blockSize\n" );
    }
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
