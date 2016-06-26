#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#define MIN_LIM 12.0
#define MAX_LIM 30.0
#define LSIZE 31 //Fixed line size in bytes

void check_input(int argc,char *argv[]);
long calc_time(struct timespec start, struct timespec end, char print_flag);
long calc_bytes(char *filename);

int main(int argc,char * argv[])
{
	check_input(argc,argv);												// Simple argument number checking.
	int rank,agents,err=0; 												// Declaration of variables
	long coords_total,total_within,byte_count,time_elapsed,total;		// used independendtly on processes.
	struct timespec start, end; 									// Variables used for time calculation.
	if(!MPI_Init(&argc,&argv))
	{
		MPI_Comm_size(MPI_COMM_WORLD,&agents); 			// Initialize OpenMPI constants
		MPI_Comm_rank(MPI_COMM_WORLD,&rank); 				// world size and rank numbers
		// Initialize vars needed for configuration
		//int runtime = atoi(argv[2]);
		int threads_num = atoi(argv[4]);
		char *file = argv[3];
		long coords_within_lim = 0;
		int proc_num = atoi(argv[5]);						// Define and handle
		if (proc_num>agents || proc_num==-1 )		// number of processes argument
		{
			proc_num=agents; 											// Set number of processes the same as mpirun/mpiexec argument
		}
		if(rank<proc_num) 											// Ignore redundant processes
		{
			float coords_val[3] = {0, 0, 0};			// Matrix stores read coordinates for from file
			if (rank==0) 													// Operations to be done only from 1 process
			{																			// to avoid	redundant delays
				time_elapsed = 0;
				int coll = atoi(argv[1]);
				// Handle max_collisions argument
				if(coll==-1)
				{
					printf("[!] Setting the number of collisions to the maximum (taken from input file)\n");
					byte_count = calc_bytes(file);						// Count the lines of input file
				}
				else
				{
					byte_count = coll*LSIZE;
				}
				if(threads_num > omp_get_max_threads() || threads_num==-1)
				{
					printf("[!] Setting the number of threads to the maximum available\n");
					omp_set_dynamic(0);
					omp_set_num_threads(omp_get_max_threads());
					threads_num=omp_get_max_threads();
				}
				else
				{
					omp_set_dynamic(0);
					omp_set_num_threads(threads_num);
				}
				clock_gettime(CLOCK_MONOTONIC, &start);										// Initialize time calculation
			}
			MPI_Bcast(&byte_count,1,MPI_LONG,0,MPI_COMM_WORLD);					// Sent only the necessary data
			MPI_Bcast(&threads_num,1,MPI_LONG,0,MPI_COMM_WORLD);				// to other processes
			char *fromfile;
			int input = open(file,O_RDONLY,S_IREAD);			// File opening
			if(input==-1)
			{
				printf("[!] Input file does not exist.\nExiting...\n");
				MPI_Finalize();
			}
			long totallines=byte_count/LSIZE;//aprox 10Mb per process
			long chunk=totallines/10;
			long loop_count=totallines/chunk;
			long offset;
			long linesperproc=chunk/proc_num;
			long finale=0;
			if (rank==proc_num-1) linesperproc+=chunk%proc_num;
			fromfile=malloc(linesperproc*LSIZE);
			int k;
			coords_within_lim=0;
			total=0;
		for (k=0;k<loop_count;k++)
		{
			offset=rank*(chunk/proc_num)*LSIZE+finale*LSIZE;
			if(rank==proc_num-1 )
			{
				finale+=(chunk/proc_num)*(proc_num-1)+linesperproc;
			}
			long max_mem=linesperproc*LSIZE;
			pread(input,fromfile,max_mem,offset);//read max_mem bytes from file and store to variable
			char *threadarray;
			#pragma omp parallel shared(fromfile,input) private(threadarray,coords_val) num_threads(threads_num)
			{
					int j,i;
					long linesperthread=linesperproc/threads_num;						//lines per thread
					long thesi=omp_get_thread_num()*linesperthread*LSIZE;				//from where every thread starts processing
					if (omp_get_thread_num()==threads_num-1) linesperthread+=linesperproc%threads_num;
					threadarray=&fromfile[thesi];
					for(j=0;j<linesperthread*LSIZE;j+=LSIZE)
					{
						for(i=0;i<3;i++)
						{
							int w;char temp[9];
							for (w=0;w<9;w++)
							{
								temp[w]=threadarray[j+i*10+w];
							}
							coords_val[i]=atof(temp);
						}

						if(coords_val[0] >= MIN_LIM && coords_val[0] <= MAX_LIM && coords_val[1] >= MIN_LIM && coords_val[1] <= MAX_LIM && coords_val[2] >= MIN_LIM && coords_val[2] <= MAX_LIM)
						{
							#pragma omp critical
							coords_within_lim++;		// If the current coordinate is within the accepted limits, update the number of accepted coordinates
						}
					 #pragma omp atomic
					 total++;

				}
					#pragma omp barrier
			}
			#pragma omp barrier
			MPI_Barrier(MPI_COMM_WORLD);
		 MPI_Bcast(&finale,1,MPI_LONG,proc_num-1,MPI_COMM_WORLD);
		}
				close(input);
				free(fromfile);
		}
		MPI_Reduce(&coords_within_lim,&total_within,1,MPI_LONG,MPI_SUM,0,MPI_COMM_WORLD); //Sum all coordinates within limit of interest
		MPI_Reduce(&total,&coords_total,1,MPI_LONG,MPI_SUM,0,MPI_COMM_WORLD);
		if(rank==0)
		{
			clock_gettime(CLOCK_MONOTONIC, &end);		// Stop the timer
			time_elapsed = calc_time(start, end, 1);	// Calculate the time elapsed
			printf("[+] %ld coordinates have been read\n[+] %ld cooordinates were inside the area of interest\n[+] %ld coordinates read per second\n",coords_total, total_within,coords_total/time_elapsed);
			printf("[+] Total Processes: %d\n",proc_num );
			printf("[+] Threads: %d\n",threads_num );
		}
		MPI_Finalize();
	}
	else
	{
			MPI_Abort(MPI_COMM_WORLD,err); 																//Abort OMPI parallel operation
	}
	return 0;
}

void check_input(int argc,char *argv[]) 																		// Handle number of arguments errors and show usage
{
	if (argc<6 || argc>6)
	{
		printf("[-] Usage: ./examine [max_collisions] [max_exec_time] [input_file] [Threads] [Processes] \nUse \"-1\": for no boundies \n");
		if (argc==2) if (!strcmp(argv[1],"--help")) printf("max_collisions: Maximum number of collisions\nmax_exec_time: Maximum execution time\ninput_file: Filename to examine\nThreads: Number of threads to use\nProcesses: Number of Processes to use.\n" );
		exit(2);
	}
}

long calc_time(struct timespec start, struct timespec end, char print_flag)	// Function that calculates the time elapsed between start - end
{																																						// Returns the time elapsed in seconds for program handling (2nd arg)
	long interval_sec = end.tv_sec - start.tv_sec;														// print_flag is a flag that enables printing
	long interval_nsec = end.tv_nsec - start.tv_nsec;
	if(interval_nsec < 0)
	{
		interval_nsec += 1000000000;
		interval_sec--;
	}
	if(print_flag == 1)
	{
		printf("[+] Main part of the program was being executed for :: %ld.%06ld :: sec\n", interval_sec, interval_nsec);
	}
	return interval_sec;
}

long calc_bytes(char *filename) 																						// Calculates the lines of input file
{
	FILE *file=fopen(filename,"r");
	fseek(file,0L,SEEK_END);															//set file position indicator right to the end-of-file
	long bytes=ftell(file);																//store the number of bytes since the beginning of the file
	fseek(file,0L,SEEK_SET);
	fclose(file);
	return bytes;																			//return lines count of the file
}
