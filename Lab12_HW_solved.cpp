#include <thrust/for_each.h>
#include <thrust/transform_reduce.h>
#include <thrust/extrema.h>
#include <thrust/inner_product.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/sequence.h>
#include <thrust/device_vector.h>
#include <thrust/count.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/constant_iterator.h>

#include <iostream>
#include <chrono>
#include <algorithm>
#include <random>



int main(void) {
	std::chrono::time_point<std::chrono::system_clock> start, end;
	std::chrono::duration<double> elapsed_seconds;

	std::mt19937 gen(0);
  std::normal_distribution<> d(500,200);
	const int N = 1<<20;
	std::vector<int> inputdata(N);
	std::generate(inputdata.begin(), inputdata.end(),[&d,&gen](){return std::max(0,(int)std::round(d(gen)));});
	std::vector<int> inputkeys(N);
	std::generate(inputkeys.begin(), inputkeys.end(),[&d,&gen](){return gen()%20;});


	/*You are given a vector of scores between 0 and 1000 (inputdata)
	  Copy those values into a thrust::device_vector and compute
	  the average of scores (~500), and their standard deviation (~200).
	  Make sure that you return and calculate with double values
	  instead of ints, otherwise you will get integer overflow problems.
	  Hint: for st. dev. use thrust::transform_reduce
	  where the operators take integers but return doubles
	  Calculate the number of scores above 800, 500, 200.
	  Find the minimum an maximum element. Try using thrust::minmax_element!
	  Hint: use "auto" for the return type
	  Check if there is an element with value 1234
	*/
	{
		thrust::device_vector<int> d_inputdata(inputdata);
		thrust::device_vector<double> d_avg(N);


		double avg = thrust::transform_reduce(d_inputdata.begin(), d_inputdata.end(), thrust::negate<int>(), 0.0, thrust::plus<int>())/-N;
		std::cout << "AVG: " << avg << std::endl;
		thrust::fill(d_avg.begin(), d_avg.end(), avg);
		
		thrust::transform(d_inputdata.begin(), d_inputdata.end(), d_avg.begin(), d_avg.begin(), thrust::minus<double>());
		double SD = std::sqrt(thrust::transform_reduce(d_avg.begin(), d_avg.end(), thrust::square<int>(), 0.0, thrust::plus<double>())/(N-1));
		std::cout << "SD: " << SD << std::endl;		

		int a200 = thrust::count_if(thrust::device, d_inputdata.begin(), d_inputdata.end(), [](const int x){return x > 200;});
		std::cout << "ABOVE 200: " << a200 << std::endl;

		int a500 = thrust::count_if(thrust::device, d_inputdata.begin(), d_inputdata.end(), [](const int x){return x > 500;});
		std::cout << "ABOVE 500: " << a500 << std::endl;

		int a800 = thrust::count_if(thrust::device, d_inputdata.begin(), d_inputdata.end(), [](const int x){return x > 800;});
		std::cout << "ABOVE 800: " << a800 << std::endl;

		thrust::device_vector<int>::iterator iter = thrust::min_element(d_inputdata.begin(), d_inputdata.end());
		std::cout << "MIN: " << *iter << std::endl;		

		thrust::device_vector<int>::iterator iter2 = thrust::max_element(d_inputdata.begin(), d_inputdata.end());
		std::cout << "MAX: " << *iter2 << std::endl;	

		int check1234 = thrust::count_if(thrust::device, d_inputdata.begin(), d_inputdata.end(), [](const int x){return x == 1234;});
		std::cout << "1234 PRESENT: " << (check1234 >= 1 ? "YES" : "NO") << std::endl;	
		
	}

	/* You are given a vector of scores between 0 and 1000 (inputdata)
	 * and each score belongs to one of 20 groups (intputkeys).
	 * Copy both vectors into thrust::device_vectors
	 * Calculate the sum of scores of each group.
	 * Hint: use reduce_by_key, which requires sorted keys.
	 * Next you will calculate the number of scores in each group
	 * by using reduce_by_key again, but instead of reducing
	 * scores, you reduce only values "1"
	 * Hint: use a thrust::constant_iterator for that
	 * Finally, compute the average scores of each group
	 * Hint use thrust::transform with two inputs and one output
	 */
	{
		std::cout << std::endl;
		thrust::device_vector<int> d_intputkeys(inputkeys);
		thrust::device_vector<int> d_inputdata(inputdata);
		
		int numKeys = 20;
		thrust::device_vector<int> d_datares(numKeys);
		thrust::device_vector<int> d_keyres(numKeys);
		
		thrust::sort_by_key(d_intputkeys.begin(), d_intputkeys.end(), d_inputdata.begin(), thrust::less<int>());
		thrust::reduce_by_key(d_intputkeys.begin(), d_intputkeys.end(), d_inputdata.begin(),d_keyres.begin(), d_datares.begin());

		for (int i = 0; i < numKeys; i++)
			std::cout << "Group " << d_keyres[i] << " sum: " << d_datares[i] << std::endl;

		std::cout << std::endl;

		thrust::device_vector<int> d_dataresC(numKeys);
		thrust::device_vector<int> d_keyresC(numKeys);

		thrust::reduce_by_key(d_intputkeys.begin(), d_intputkeys.end(), thrust::make_constant_iterator(1),d_keyresC.begin(), d_dataresC.begin(),thrust::equal_to<int>(), thrust::plus<int>());

		for (int i = 0; i < numKeys; i++)
			std::cout << "Group " << d_keyresC[i] << " count: " << d_dataresC[i] << std::endl;

		std::cout << std::endl;

		thrust::device_vector<double> d_avgres(numKeys);
		thrust::transform(d_datares.begin(), d_datares.end(), d_dataresC.begin(), d_avgres.begin(), thrust::divides<double>());

		for (int i = 0; i < numKeys; i++)
			std::cout << "Group " << d_keyresC[i] << " avg: " << d_avgres[i] << std::endl;

	}

	/* Copy both arrays to the device again, and create a separate
	 * array with an index for each score (0->N-1).
	 * Sort the scores in descending order, along with the group
	 * and index values
	 * Hint: use make_zip_iterator to zip groups and indices
	 * From the best 20 scores, select the ones that are in different
	 * groups. Hint: use unique, with a zip iterator and with all
	 * 3 arrays in a single tuple. Keep in mind, that unique expects
	 * a sorted input (by group in this case).
	 * Print the indices, groups and scores of these
	 */
	{
		int topCount = 20;

		std::cout << std::endl;
		thrust::device_vector<int> d_intputkeys(inputkeys);
		thrust::device_vector<int> d_inputdata(inputdata);

		thrust::device_vector<int> d_idx(inputdata.size());
		thrust::sequence(d_idx.begin(), d_idx.end());

		thrust::sort_by_key(d_inputdata.begin(), d_inputdata.end(), thrust::make_zip_iterator(thrust::make_tuple(d_intputkeys.begin(), d_idx.begin())), thrust::greater<int>());

		thrust::device_vector<int> d_intputkeysRes(topCount);
		thrust::device_vector<int> d_inputdataRes(topCount);

		thrust::device_vector<int> d_idxRes(topCount);

		thrust::copy(thrust::device, d_intputkeys.begin(), d_intputkeys.begin() + topCount, d_intputkeysRes.begin());
		thrust::copy(thrust::device, d_inputdata.begin(), d_inputdata.begin() + topCount, d_inputdataRes.begin());
		thrust::copy(thrust::device, d_idx.begin(), d_idx.begin() + topCount, d_idxRes.begin());


		thrust::sort_by_key(d_intputkeysRes.begin(), d_intputkeysRes.end(), thrust::make_zip_iterator(thrust::make_tuple(d_inputdataRes.begin(), d_idxRes.begin())), thrust::less<int>());
		auto end = thrust::unique(
					d_intputkeysRes.begin(),
					d_intputkeysRes.end()
			);

		d_intputkeysRes.erase(end, d_intputkeysRes.end());
		d_inputdataRes.erase(d_inputdataRes.begin() + d_intputkeysRes.size()-1, d_inputdataRes.end());
		d_idxRes.erase(d_idxRes.begin() + d_intputkeysRes.size()-1, d_idxRes.end());;

		for (int i = 0; i < d_idxRes.size(); i++)
			std::cout << "Group " << d_intputkeysRes[i] << ", idx: " << d_idxRes[i] << ", score: " <<  d_inputdataRes[i] << std::endl;
	}

	return 0;
}
