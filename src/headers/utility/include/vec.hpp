#ifndef VEC_HPP
#define VEC_HPP

#include "vec.h"


template<int n, typename T>
__DEVICE__ __HOST__ Vec<n, T>::Vec(){
	for(auto i=0; i<n; ++i)
		data[i] = 0.0;
}

template<int n, typename T>
template <typename ...Args>
__DEVICE__ __HOST__ Vec<n, T>::Vec(Args... args){
	auto sz = sizeof...(args);
	if(sz <= n){
		auto dp = data;
		for(auto val: {args...})
			*(dp++) = val;
		for(auto i=sz; i<n; ++i)
			*(dp++) = 0.0;
	}
	else{
		auto dp = data;
		for(auto val: {args...})
			if(dp-data < n)
				*(dp++) = val;
	}
}

template<int n, typename T>
__DEVICE__ __HOST__ Vec<n, T>::Vec(T* const &t){
	for(auto i=0; i<n; ++i)
		data[i] = t[i];
}

template<int n, typename T>
__DEVICE__ __HOST__ T& Vec<n, T>::operator[](int const &index){
	return data[index];
}

template<int n, typename T>
__DEVICE__ __HOST__ T Vec<n, T>::operator[](int const &index) const{
	return data[index];
}


/*** operator overrides ***/

template<int n, typename T>
__DEVICE__ __HOST__ Vec<n, T> Vec<n, T>::operator*(T const &d) const{
	T tmp[n];
	for(auto i=0; i<n; ++i)
		tmp[i] = data[i] *d;
	return Vec<n, T>{tmp};
}

template<int n, typename T>
__DEVICE__ __HOST__ Vec<n, T> Vec<n, T>::operator/(T const &d) const{
	T tmp[n];
	for(auto i=0; i<n; ++i)
		tmp[i] = data[i] /d;
	return Vec<n, T>{tmp};
}

template<int n, typename T>
__DEVICE__ __HOST__ Vec<n, T> Vec<n, T>::operator+=(Vec<n, T> const &a){
	return *this = *this + a;
}

template<int n, typename T>
__DEVICE__ __HOST__ Vec<n, T> Vec<n, T>::operator-=(Vec<n, T> const &a){
	return *this = *this - a;
}

template<int n, typename T>
__DEVICE__ __HOST__ Vec<n, T> Vec<n, T>::operator*=(T const &d){
	return *this = *this * d;
}

template<int n, typename T>
__DEVICE__ __HOST__ Vec<n, T> Vec<n, T>::operator/=(T const &d){
	return *this = *this / d;
}

//end of class definitions


/*** unsymmetric operators ***/
template<int n, typename T>
__DEVICE__ __HOST__ Vec<n, T> operator+(Vec<n, T> const &a, Vec<n, T> const &b){
	T tmp[n];
	for(auto i=0; i<n; ++i)
		tmp[i] = a[i] + b[i];
	return Vec<n, T>{tmp};
}

template<int n, typename T>
__DEVICE__ __HOST__ Vec<n, T> operator-(Vec<n, T> const &a, Vec<n, T> const &b){
	T tmp[n];
	for(auto i=0; i<n; ++i)
		tmp[i] = a[i] - b[i];
	return Vec<n, T>{tmp};
}

template<int n, typename T>
__DEVICE__ __HOST__ T operator*(Vec<n, T> const &a, Vec<n, T> const &b){
	T ret;
	for(auto i=0; i<n; ++i)
		ret += a[i] * b[i];
	return ret;
}

template<int n, typename T>
__DEVICE__ __HOST__ Vec<n, T> operator*(T const &d, Vec<n, T> const &b){
	return b * d;
}

/*** math transform functions ***/
template<int n, typename T>
__DEVICE__ __HOST__ T length(Vec<n, T> const &a){
	return sqrt(a*a);
}

template<int n, typename T>
__DEVICE__ __HOST__ Vec<n, T> norm(Vec<n, T> const &a){
	auto len = length(a);
	if(len != 0.0)
		return a/len;
	else{
		printf("Error when norm vec: divided by zero.\n");
		return {0.0, 0.0};
	}
}

/*** vector printer  ***/
template<int n, typename T>
__DEVICE__ __HOST__ std::ostream& operator<<(std::ostream &os, Vec<n, T> const &vec){
	for(auto i=0; i<n; ++i)
		os << vec[i] << "\t";
	return os;
}

#endif