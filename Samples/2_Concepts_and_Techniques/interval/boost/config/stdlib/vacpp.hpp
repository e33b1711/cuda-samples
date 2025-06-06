//  (C) Copyright John Maddock 2001 - 2002.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//  See http://www.boost.org for most recent version.

#if __IBMCPP__ <= 501
#define BOOST_NO_STD_ALLOCATOR
#endif

#define BOOST_HAS_MACRO_USE_FACET
#define BOOST_NO_STD_MESSAGES

//  C++0x headers not yet implemented
//
#define BOOST_NO_0X_HDR_ARRAY
#define BOOST_NO_0X_HDR_CHRONO
#define BOOST_NO_0X_HDR_CODECVT
#define BOOST_NO_0X_HDR_CONCEPTS
#define BOOST_NO_0X_HDR_CONDITION_VARIABLE
#define BOOST_NO_0X_HDR_CONTAINER_CONCEPTS
#define BOOST_NO_0X_HDR_FORWARD_LIST
#define BOOST_NO_0X_HDR_FUTURE
#define BOOST_NO_0X_HDR_INITIALIZER_LIST
#define BOOST_NO_0X_HDR_ITERATOR_CONCEPTS
#define BOOST_NO_0X_HDR_MEMORY_CONCEPTS
#define BOOST_NO_0X_HDR_MUTEX
#define BOOST_NO_0X_HDR_RANDOM
#define BOOST_NO_0X_HDR_RATIO
#define BOOST_NO_0X_HDR_REGEX
#define BOOST_NO_0X_HDR_SYSTEM_ERROR
#define BOOST_NO_0X_HDR_THREAD
#define BOOST_NO_0X_HDR_TUPLE
#define BOOST_NO_0X_HDR_TYPE_TRAITS
#define BOOST_NO_STD_UNORDERED // deprecated; see following
#define BOOST_NO_0X_HDR_UNORDERED_MAP
#define BOOST_NO_0X_HDR_UNORDERED_SET

#define BOOST_STDLIB "Visual Age default standard library"
