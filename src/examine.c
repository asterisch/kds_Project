#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>

#define MIN_LIM 12.0
#define MAX_LIM 30.0
#define LSIZE 31 //Fixed line size in bytes

void check_input(int argc,char *argv[]);
long calc_time(struct timespec start, struct timespec end, char print_flag);
long calc_lines(char *filename);

int main(int argc,char * argv[])
{
	check_input(argc,argv);					// Simple argument number checking
	int rank,agents,err=0;
	long coords_total,loop_count,time_elapsed;
	if(!MPI_Init(&argc,&argv))
	{
		MPI_Comm_size(MPI_COMM_WORLD,&agents); // Initialize OpenMPI constants
		MPI_Comm_rank(MPI_COMM_WORLD,&rank); // world size and rank numbers
		struct timespec start, end;				// Initialize vars needed for configuration
		//int runtime = atoi(argv[2]);			// and basic calculations
		int threads_num = atoi(argv[4]);		// in order to get
		char *file = argv[3];
		long coords_within_lim = 0;
		int proc_num = atoi(argv[5]);			// the desired output
		if (proc_num>agents || proc_num==-1 ) proc_num=agents;
		if(rank<proc_num)
		{
			float coords_val[3] = {0, 0, 0};
			FILE *input = fopen(file, "r");			// File desccriptor for every process
			if(!input)
			{
					printf("[!] Input file does not exist.\nExiting...\n");
					exit(3);
			}
			if (rank==0)
			{
				time_elapsed = 0;
				loop_count = calc_lines(file);
				clock_gettime(CLOCK_MONOTONIC, &start);		// Initialize time calculation
				int coll = atoi(argv[1]);
				if(coll != -1)
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
				if(threads_num != -1)
				{
					if(threads_num > omp_get_max_threads())
					{
						printf("[!] Warning: Specified threads exceed the number of available ones\n");
						printf("[!] Setting the number of threads to the maximum available");
						omp_set_dynamic(0);
						omp_set_num_threads(omp_get_max_threads());
					}
					else
					{
						omp_set_num_threads(threads_num);
					}
				}
			}
			MPI_Bcast(&loop_count,1,MPI_LONG,0,MPI_COMM_WORLD);
			long loadperproc=loop_count/proc_num;							//Asign corresponding load at every process
			fseek(input,rank*loadperproc*LSIZE,SEEK_SET);  //Move the file position indicator of every process
			if(rank==proc_num-1) loadperproc+=(loop_count%proc_num); //Increment load of last process so as to reach end-of-file manually
			int i;
			#pragma omp parallel for shared(loadperproc, coords_within_lim, input) private(coords_val, i) schedule(dynamic)
			for(i=0; i<loadperproc; i++)					// The main loop of the program
			{
				fscanf(input, "%f %f %f", &coords_val[0], &coords_val[1], &coords_val[2]);
				if(coords_val[0] >= MIN_LIM && coords_val[0] <= MAX_LIM && coords_val[1] >= MIN_LIM && coords_val[1] <= MAX_LIM && coords_val[2] >= MIN_LIM && coords_val[2] <= MAX_LIM)
				{
					#pragma omp atomic
					coords_within_lim++;		// If the current coordinate is within the accepted limits, update the number of accepted coordinates
				}
			}
			fclose(input); //Close file of every process
			printf("Rank %d found: %ld coords in limits\n",rank,coords_within_lim);
		}
		MPI_Reduce(&coords_within_lim,&coords_total,1,MPI_LONG,MPI_SUM,0,MPI_COMM_WORLD); //Sum all coordinates within limit of interest
		if(rank==0)
		{
			clock_gettime(CLOCK_MONOTONIC, &end);		// Stop the timer
			time_elapsed = calc_time(start, end, 1);	// Calculate the time elapsed
			printf("[+] %ld coordinates have been read\n[+] %ld cooordinates were inside the area of interest\n[+] %ld coordinates read per second\n", loop_count, coords_total, loop_count/time_elapsed);
			printf("[+] Total Processes: %d\n",proc_num );
		}
		MPI_Finalize();
	}
	else
	{
			MPI_Abort(MPI_COMM_WORLD,err); 		//Abort OMPI parallel operation
	}
	return 0;
}

void check_input(int argc,char *argv[])
{
	if (argc<6 || argc>6) // If input arguments are not as much as meant to be, the algorithm ends indicating it's usage.
	{
		printf("[-] Usage: ./examine [max_collisions] [max_exec_time] [input_file] [Threads] [Processes] \nUse \"-1\": for no boundies \n");
		if (argc==2) if (!strcmp(argv[1],"--help")) printf("max_collisions: Maximum number of collisions\nmax_exec_time: Maximum execution time\ninput_file: Filename to examine\nThreads: Number of threads to use\nProcesses: Number of Processes to use.\n" );
		exit(2);
	}
}

long calc_time(struct timespec start, struct timespec end, char print_flag)	// Function that calculates the time elapsed between start - end
{																			// Returns the time elapsed in seconds for program handling (2nd arg)
	long interval_sec = end.tv_sec - start.tv_sec;							// print_flag is a flag that enables printing
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

long calc_lines(char *filename)
{
	FILE *file=fopen(filename,"r");
	fseek(file,0L,SEEK_END);	//set file position indicator right to the end-of-file
	long lines=ftell(file);		//store the number of bytes since the beginning of the file
	fseek(file,0L,SEEK_SET);
	fclose(file);
	return lines/31;		//return lines count of the file
}
