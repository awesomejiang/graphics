#ifndef VEC_H
#define VEC_H

#include <cuda.h>

#include <iostream>

#include "macros.h"

//TODO:
//1. add some enable_if checks
//2. can we add "templated enums" for indexing data[...]: {x,y}, {r,g,b,a}...

template<int n, typename T>
class Vec{
public:
	__DEVICE__ __HOST__ Vec();

	template <typename ...Args>
	__DEVICE__ __HOST__ Vec(Args... args);

	__DEVICE__ __HOST__ Vec(T* const &t);

	__DEVICE__ __HOST__ T& operator[](int const &index);
	__DEVICE__ __HOST__ T operator[](int const &index) const;

	//operator overrides
	__DEVICE__ __HOST__ Vec<n, T> operator*(T const &d) const;
	__DEVICE__ __HOST__ Vec<n, T> operator/(T const &d) const;
	__DEVICE__ __HOST__ Vec<n, T> operator+=(Vec<n, T> const &a);
	__DEVICE__ __HOST__ Vec<n, T> operator-=(Vec<n, T> const &a);
	__DEVICE__ __HOST__ Vec<n, T> operator*=(T const &d);
	__DEVICE__ __HOST__ Vec<n, T> operator/=(T const &d);

	T data[n];
};

//unsymmetric operators
template<int n, typename T>
__DEVICE__ __HOST__ Vec<n, T> operator+(Vec<n, T> const &a, Vec<n, T> const &b);

template<int n, typename T>
__DEVICE__ __HOST__ Vec<n, T> operator-(Vec<n, T> const &a, Vec<n, T> const &b);

template<int n, typename T>
__DEVICE__ __HOST__ T operator*(Vec<n, T> const &a, Vec<n, T> const &b);

template<int n, typename T>
__DEVICE__ __HOST__ Vec<n, T> operator*(T const &d, Vec<n, T> const &b);

//math transform functions
template<int n, typename T>
__DEVICE__ __HOST__ T length(Vec<n, T> const &a);

template<int n, typename T>
__DEVICE__ __HOST__ Vec<n, T> norm(Vec<n, T> const &a);

//print vector
template<int n, typename T>
__DEVICE__ __HOST__ std::ostream& operator<<(std::ostream &os, Vec<n, T> const &vec);

#endif