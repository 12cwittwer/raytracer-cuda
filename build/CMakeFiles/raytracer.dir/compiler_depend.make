# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

CMakeFiles/raytracer.dir/src/cpu/main.cpp.o: /home/christianw/raytracer-cuda/src/cpu/main.cpp \
  /home/christianw/raytracer-cuda/include/raytracer/aabb.h \
  /home/christianw/raytracer-cuda/include/raytracer/bvh.h \
  /home/christianw/raytracer-cuda/include/raytracer/bvh_builder.h \
  /home/christianw/raytracer-cuda/include/raytracer/camera.h \
  /home/christianw/raytracer-cuda/include/raytracer/camera_data.h \
  /home/christianw/raytracer-cuda/include/raytracer/color.h \
  /home/christianw/raytracer-cuda/include/raytracer/cuda_compat.h \
  /home/christianw/raytracer-cuda/include/raytracer/cuda_utils.h \
  /home/christianw/raytracer-cuda/include/raytracer/hittable.h \
  /home/christianw/raytracer-cuda/include/raytracer/hittable_dispatch.h \
  /home/christianw/raytracer-cuda/include/raytracer/hittable_dispatch_impl.h \
  /home/christianw/raytracer-cuda/include/raytracer/hittable_list.h \
  /home/christianw/raytracer-cuda/include/raytracer/instances.h \
  /home/christianw/raytracer-cuda/include/raytracer/interval.h \
  /home/christianw/raytracer-cuda/include/raytracer/material.h \
  /home/christianw/raytracer-cuda/include/raytracer/quad.h \
  /home/christianw/raytracer-cuda/include/raytracer/random_utils.h \
  /home/christianw/raytracer-cuda/include/raytracer/ray.h \
  /home/christianw/raytracer-cuda/include/raytracer/render.h \
  /home/christianw/raytracer-cuda/include/raytracer/rtweekend.h \
  /home/christianw/raytracer-cuda/include/raytracer/sphere_gpu.h \
  /home/christianw/raytracer-cuda/include/raytracer/vec3.h \
  /usr/include/alloca.h \
  /usr/include/asm-generic/errno-base.h \
  /usr/include/asm-generic/errno.h \
  /usr/include/builtin_types.h \
  /usr/include/c++/13/algorithm \
  /usr/include/c++/13/backward/binders.h \
  /usr/include/c++/13/bit \
  /usr/include/c++/13/bits/algorithmfwd.h \
  /usr/include/c++/13/bits/alloc_traits.h \
  /usr/include/c++/13/bits/allocator.h \
  /usr/include/c++/13/bits/basic_ios.h \
  /usr/include/c++/13/bits/basic_ios.tcc \
  /usr/include/c++/13/bits/basic_string.h \
  /usr/include/c++/13/bits/basic_string.tcc \
  /usr/include/c++/13/bits/char_traits.h \
  /usr/include/c++/13/bits/charconv.h \
  /usr/include/c++/13/bits/concept_check.h \
  /usr/include/c++/13/bits/cpp_type_traits.h \
  /usr/include/c++/13/bits/cxxabi_forced.h \
  /usr/include/c++/13/bits/cxxabi_init_exception.h \
  /usr/include/c++/13/bits/exception.h \
  /usr/include/c++/13/bits/exception_defines.h \
  /usr/include/c++/13/bits/exception_ptr.h \
  /usr/include/c++/13/bits/functexcept.h \
  /usr/include/c++/13/bits/functional_hash.h \
  /usr/include/c++/13/bits/hash_bytes.h \
  /usr/include/c++/13/bits/invoke.h \
  /usr/include/c++/13/bits/ios_base.h \
  /usr/include/c++/13/bits/istream.tcc \
  /usr/include/c++/13/bits/locale_classes.h \
  /usr/include/c++/13/bits/locale_classes.tcc \
  /usr/include/c++/13/bits/locale_facets.h \
  /usr/include/c++/13/bits/locale_facets.tcc \
  /usr/include/c++/13/bits/localefwd.h \
  /usr/include/c++/13/bits/memory_resource.h \
  /usr/include/c++/13/bits/memoryfwd.h \
  /usr/include/c++/13/bits/move.h \
  /usr/include/c++/13/bits/nested_exception.h \
  /usr/include/c++/13/bits/new_allocator.h \
  /usr/include/c++/13/bits/ostream.tcc \
  /usr/include/c++/13/bits/ostream_insert.h \
  /usr/include/c++/13/bits/postypes.h \
  /usr/include/c++/13/bits/predefined_ops.h \
  /usr/include/c++/13/bits/ptr_traits.h \
  /usr/include/c++/13/bits/range_access.h \
  /usr/include/c++/13/bits/refwrap.h \
  /usr/include/c++/13/bits/requires_hosted.h \
  /usr/include/c++/13/bits/specfun.h \
  /usr/include/c++/13/bits/std_abs.h \
  /usr/include/c++/13/bits/stl_algo.h \
  /usr/include/c++/13/bits/stl_algobase.h \
  /usr/include/c++/13/bits/stl_bvector.h \
  /usr/include/c++/13/bits/stl_construct.h \
  /usr/include/c++/13/bits/stl_function.h \
  /usr/include/c++/13/bits/stl_heap.h \
  /usr/include/c++/13/bits/stl_iterator.h \
  /usr/include/c++/13/bits/stl_iterator_base_funcs.h \
  /usr/include/c++/13/bits/stl_iterator_base_types.h \
  /usr/include/c++/13/bits/stl_pair.h \
  /usr/include/c++/13/bits/stl_relops.h \
  /usr/include/c++/13/bits/stl_tempbuf.h \
  /usr/include/c++/13/bits/stl_uninitialized.h \
  /usr/include/c++/13/bits/stl_vector.h \
  /usr/include/c++/13/bits/streambuf.tcc \
  /usr/include/c++/13/bits/streambuf_iterator.h \
  /usr/include/c++/13/bits/string_view.tcc \
  /usr/include/c++/13/bits/stringfwd.h \
  /usr/include/c++/13/bits/uniform_int_dist.h \
  /usr/include/c++/13/bits/uses_allocator.h \
  /usr/include/c++/13/bits/uses_allocator_args.h \
  /usr/include/c++/13/bits/utility.h \
  /usr/include/c++/13/bits/vector.tcc \
  /usr/include/c++/13/cctype \
  /usr/include/c++/13/cerrno \
  /usr/include/c++/13/clocale \
  /usr/include/c++/13/cmath \
  /usr/include/c++/13/cstddef \
  /usr/include/c++/13/cstdio \
  /usr/include/c++/13/cstdlib \
  /usr/include/c++/13/cwchar \
  /usr/include/c++/13/cwctype \
  /usr/include/c++/13/debug/assertions.h \
  /usr/include/c++/13/debug/debug.h \
  /usr/include/c++/13/exception \
  /usr/include/c++/13/ext/alloc_traits.h \
  /usr/include/c++/13/ext/atomicity.h \
  /usr/include/c++/13/ext/numeric_traits.h \
  /usr/include/c++/13/ext/string_conversions.h \
  /usr/include/c++/13/ext/type_traits.h \
  /usr/include/c++/13/initializer_list \
  /usr/include/c++/13/ios \
  /usr/include/c++/13/iosfwd \
  /usr/include/c++/13/iostream \
  /usr/include/c++/13/istream \
  /usr/include/c++/13/limits \
  /usr/include/c++/13/math.h \
  /usr/include/c++/13/new \
  /usr/include/c++/13/ostream \
  /usr/include/c++/13/pstl/execution_defs.h \
  /usr/include/c++/13/pstl/glue_algorithm_defs.h \
  /usr/include/c++/13/pstl/pstl_config.h \
  /usr/include/c++/13/stdexcept \
  /usr/include/c++/13/stdlib.h \
  /usr/include/c++/13/streambuf \
  /usr/include/c++/13/string \
  /usr/include/c++/13/string_view \
  /usr/include/c++/13/system_error \
  /usr/include/c++/13/tr1/bessel_function.tcc \
  /usr/include/c++/13/tr1/beta_function.tcc \
  /usr/include/c++/13/tr1/ell_integral.tcc \
  /usr/include/c++/13/tr1/exp_integral.tcc \
  /usr/include/c++/13/tr1/gamma.tcc \
  /usr/include/c++/13/tr1/hypergeometric.tcc \
  /usr/include/c++/13/tr1/legendre_function.tcc \
  /usr/include/c++/13/tr1/modified_bessel_func.tcc \
  /usr/include/c++/13/tr1/poly_hermite.tcc \
  /usr/include/c++/13/tr1/poly_laguerre.tcc \
  /usr/include/c++/13/tr1/riemann_zeta.tcc \
  /usr/include/c++/13/tr1/special_function_util.h \
  /usr/include/c++/13/tuple \
  /usr/include/c++/13/type_traits \
  /usr/include/c++/13/typeinfo \
  /usr/include/c++/13/utility \
  /usr/include/c++/13/vector \
  /usr/include/channel_descriptor.h \
  /usr/include/crt/host_config.h \
  /usr/include/crt/host_defines.h \
  /usr/include/ctype.h \
  /usr/include/cuda_device_runtime_api.h \
  /usr/include/cuda_runtime.h \
  /usr/include/cuda_runtime_api.h \
  /usr/include/curand.h \
  /usr/include/curand_discrete.h \
  /usr/include/curand_discrete2.h \
  /usr/include/curand_globals.h \
  /usr/include/curand_kernel.h \
  /usr/include/curand_lognormal.h \
  /usr/include/curand_mrg32k3a.h \
  /usr/include/curand_mtgp32.h \
  /usr/include/curand_mtgp32_kernel.h \
  /usr/include/curand_normal.h \
  /usr/include/curand_normal_static.h \
  /usr/include/curand_philox4x32_x.h \
  /usr/include/curand_poisson.h \
  /usr/include/curand_precalc.h \
  /usr/include/curand_uniform.h \
  /usr/include/device_types.h \
  /usr/include/driver_functions.h \
  /usr/include/driver_types.h \
  /usr/include/endian.h \
  /usr/include/errno.h \
  /usr/include/features-time64.h \
  /usr/include/features.h \
  /usr/include/library_types.h \
  /usr/include/limits.h \
  /usr/include/linux/errno.h \
  /usr/include/linux/limits.h \
  /usr/include/locale.h \
  /usr/include/math.h \
  /usr/include/memory.h \
  /usr/include/nv/detail/__preprocessor \
  /usr/include/nv/detail/__target_macros \
  /usr/include/nv/target \
  /usr/include/pthread.h \
  /usr/include/sched.h \
  /usr/include/stdc-predef.h \
  /usr/include/stdio.h \
  /usr/include/stdlib.h \
  /usr/include/string.h \
  /usr/include/strings.h \
  /usr/include/surface_types.h \
  /usr/include/texture_types.h \
  /usr/include/time.h \
  /usr/include/vector_functions.h \
  /usr/include/vector_functions.hpp \
  /usr/include/vector_types.h \
  /usr/include/wchar.h \
  /usr/include/wctype.h \
  /usr/include/x86_64-linux-gnu/asm/errno.h \
  /usr/include/x86_64-linux-gnu/bits/atomic_wide_counter.h \
  /usr/include/x86_64-linux-gnu/bits/byteswap.h \
  /usr/include/x86_64-linux-gnu/bits/cpu-set.h \
  /usr/include/x86_64-linux-gnu/bits/endian.h \
  /usr/include/x86_64-linux-gnu/bits/endianness.h \
  /usr/include/x86_64-linux-gnu/bits/errno.h \
  /usr/include/x86_64-linux-gnu/bits/floatn-common.h \
  /usr/include/x86_64-linux-gnu/bits/floatn.h \
  /usr/include/x86_64-linux-gnu/bits/flt-eval-method.h \
  /usr/include/x86_64-linux-gnu/bits/fp-fast.h \
  /usr/include/x86_64-linux-gnu/bits/fp-logb.h \
  /usr/include/x86_64-linux-gnu/bits/iscanonical.h \
  /usr/include/x86_64-linux-gnu/bits/libc-header-start.h \
  /usr/include/x86_64-linux-gnu/bits/libm-simd-decl-stubs.h \
  /usr/include/x86_64-linux-gnu/bits/local_lim.h \
  /usr/include/x86_64-linux-gnu/bits/locale.h \
  /usr/include/x86_64-linux-gnu/bits/long-double.h \
  /usr/include/x86_64-linux-gnu/bits/math-vector.h \
  /usr/include/x86_64-linux-gnu/bits/mathcalls-helper-functions.h \
  /usr/include/x86_64-linux-gnu/bits/mathcalls-narrow.h \
  /usr/include/x86_64-linux-gnu/bits/mathcalls.h \
  /usr/include/x86_64-linux-gnu/bits/posix1_lim.h \
  /usr/include/x86_64-linux-gnu/bits/posix2_lim.h \
  /usr/include/x86_64-linux-gnu/bits/pthread_stack_min-dynamic.h \
  /usr/include/x86_64-linux-gnu/bits/pthreadtypes-arch.h \
  /usr/include/x86_64-linux-gnu/bits/pthreadtypes.h \
  /usr/include/x86_64-linux-gnu/bits/sched.h \
  /usr/include/x86_64-linux-gnu/bits/select.h \
  /usr/include/x86_64-linux-gnu/bits/setjmp.h \
  /usr/include/x86_64-linux-gnu/bits/stdint-intn.h \
  /usr/include/x86_64-linux-gnu/bits/stdio_lim.h \
  /usr/include/x86_64-linux-gnu/bits/stdlib-float.h \
  /usr/include/x86_64-linux-gnu/bits/struct_mutex.h \
  /usr/include/x86_64-linux-gnu/bits/struct_rwlock.h \
  /usr/include/x86_64-linux-gnu/bits/thread-shared-types.h \
  /usr/include/x86_64-linux-gnu/bits/time.h \
  /usr/include/x86_64-linux-gnu/bits/time64.h \
  /usr/include/x86_64-linux-gnu/bits/timesize.h \
  /usr/include/x86_64-linux-gnu/bits/timex.h \
  /usr/include/x86_64-linux-gnu/bits/types.h \
  /usr/include/x86_64-linux-gnu/bits/types/FILE.h \
  /usr/include/x86_64-linux-gnu/bits/types/__FILE.h \
  /usr/include/x86_64-linux-gnu/bits/types/__fpos64_t.h \
  /usr/include/x86_64-linux-gnu/bits/types/__fpos_t.h \
  /usr/include/x86_64-linux-gnu/bits/types/__locale_t.h \
  /usr/include/x86_64-linux-gnu/bits/types/__mbstate_t.h \
  /usr/include/x86_64-linux-gnu/bits/types/__sigset_t.h \
  /usr/include/x86_64-linux-gnu/bits/types/clock_t.h \
  /usr/include/x86_64-linux-gnu/bits/types/clockid_t.h \
  /usr/include/x86_64-linux-gnu/bits/types/cookie_io_functions_t.h \
  /usr/include/x86_64-linux-gnu/bits/types/error_t.h \
  /usr/include/x86_64-linux-gnu/bits/types/locale_t.h \
  /usr/include/x86_64-linux-gnu/bits/types/mbstate_t.h \
  /usr/include/x86_64-linux-gnu/bits/types/sigset_t.h \
  /usr/include/x86_64-linux-gnu/bits/types/struct_FILE.h \
  /usr/include/x86_64-linux-gnu/bits/types/struct___jmp_buf_tag.h \
  /usr/include/x86_64-linux-gnu/bits/types/struct_itimerspec.h \
  /usr/include/x86_64-linux-gnu/bits/types/struct_sched_param.h \
  /usr/include/x86_64-linux-gnu/bits/types/struct_timespec.h \
  /usr/include/x86_64-linux-gnu/bits/types/struct_timeval.h \
  /usr/include/x86_64-linux-gnu/bits/types/struct_tm.h \
  /usr/include/x86_64-linux-gnu/bits/types/time_t.h \
  /usr/include/x86_64-linux-gnu/bits/types/timer_t.h \
  /usr/include/x86_64-linux-gnu/bits/types/wint_t.h \
  /usr/include/x86_64-linux-gnu/bits/typesizes.h \
  /usr/include/x86_64-linux-gnu/bits/uintn-identity.h \
  /usr/include/x86_64-linux-gnu/bits/uio_lim.h \
  /usr/include/x86_64-linux-gnu/bits/waitflags.h \
  /usr/include/x86_64-linux-gnu/bits/waitstatus.h \
  /usr/include/x86_64-linux-gnu/bits/wchar.h \
  /usr/include/x86_64-linux-gnu/bits/wctype-wchar.h \
  /usr/include/x86_64-linux-gnu/bits/wordsize.h \
  /usr/include/x86_64-linux-gnu/bits/xopen_lim.h \
  /usr/include/x86_64-linux-gnu/c++/13/bits/atomic_word.h \
  /usr/include/x86_64-linux-gnu/c++/13/bits/c++allocator.h \
  /usr/include/x86_64-linux-gnu/c++/13/bits/c++config.h \
  /usr/include/x86_64-linux-gnu/c++/13/bits/c++locale.h \
  /usr/include/x86_64-linux-gnu/c++/13/bits/cpu_defines.h \
  /usr/include/x86_64-linux-gnu/c++/13/bits/ctype_base.h \
  /usr/include/x86_64-linux-gnu/c++/13/bits/ctype_inline.h \
  /usr/include/x86_64-linux-gnu/c++/13/bits/error_constants.h \
  /usr/include/x86_64-linux-gnu/c++/13/bits/gthr-default.h \
  /usr/include/x86_64-linux-gnu/c++/13/bits/gthr.h \
  /usr/include/x86_64-linux-gnu/c++/13/bits/os_defines.h \
  /usr/include/x86_64-linux-gnu/gnu/stubs-64.h \
  /usr/include/x86_64-linux-gnu/gnu/stubs.h \
  /usr/include/x86_64-linux-gnu/sys/cdefs.h \
  /usr/include/x86_64-linux-gnu/sys/select.h \
  /usr/include/x86_64-linux-gnu/sys/single_threaded.h \
  /usr/include/x86_64-linux-gnu/sys/types.h \
  /usr/lib/gcc/x86_64-linux-gnu/13/include/limits.h \
  /usr/lib/gcc/x86_64-linux-gnu/13/include/stdarg.h \
  /usr/lib/gcc/x86_64-linux-gnu/13/include/stddef.h \
  /usr/lib/gcc/x86_64-linux-gnu/13/include/syslimits.h


/usr/lib/gcc/x86_64-linux-gnu/13/include/stddef.h:

/usr/lib/gcc/x86_64-linux-gnu/13/include/stdarg.h:

/usr/include/x86_64-linux-gnu/sys/types.h:

/usr/include/x86_64-linux-gnu/sys/single_threaded.h:

/usr/include/x86_64-linux-gnu/sys/cdefs.h:

/usr/include/x86_64-linux-gnu/gnu/stubs.h:

/usr/include/x86_64-linux-gnu/c++/13/bits/os_defines.h:

/usr/include/x86_64-linux-gnu/c++/13/bits/gthr.h:

/usr/include/x86_64-linux-gnu/c++/13/bits/error_constants.h:

/usr/include/x86_64-linux-gnu/c++/13/bits/ctype_base.h:

/usr/include/x86_64-linux-gnu/c++/13/bits/cpu_defines.h:

/usr/include/x86_64-linux-gnu/c++/13/bits/c++locale.h:

/usr/include/x86_64-linux-gnu/c++/13/bits/c++config.h:

/usr/include/x86_64-linux-gnu/bits/xopen_lim.h:

/usr/include/x86_64-linux-gnu/c++/13/bits/atomic_word.h:

/usr/include/x86_64-linux-gnu/bits/wordsize.h:

/usr/include/x86_64-linux-gnu/bits/wctype-wchar.h:

/usr/include/x86_64-linux-gnu/bits/wchar.h:

/usr/include/x86_64-linux-gnu/bits/waitflags.h:

/usr/include/x86_64-linux-gnu/bits/uio_lim.h:

/usr/include/x86_64-linux-gnu/bits/uintn-identity.h:

/usr/include/x86_64-linux-gnu/bits/typesizes.h:

/usr/include/x86_64-linux-gnu/bits/types/timer_t.h:

/usr/include/x86_64-linux-gnu/bits/types/time_t.h:

/usr/include/x86_64-linux-gnu/c++/13/bits/gthr-default.h:

/usr/include/x86_64-linux-gnu/bits/types/struct_timespec.h:

/usr/include/x86_64-linux-gnu/bits/types/struct_itimerspec.h:

/usr/include/c++/13/pstl/pstl_config.h:

/usr/include/c++/13/pstl/glue_algorithm_defs.h:

/usr/include/c++/13/bits/basic_ios.tcc:

/usr/include/channel_descriptor.h:

/usr/include/c++/13/math.h:

/usr/include/c++/13/bits/ostream_insert.h:

/usr/include/x86_64-linux-gnu/bits/types/wint_t.h:

/usr/include/c++/13/limits:

/usr/include/c++/13/istream:

/usr/include/c++/13/iostream:

/usr/include/c++/13/ext/type_traits.h:

/usr/include/c++/13/ext/string_conversions.h:

/usr/include/x86_64-linux-gnu/bits/types/sigset_t.h:

/usr/include/x86_64-linux-gnu/bits/types/struct_tm.h:

/usr/include/c++/13/ext/atomicity.h:

/usr/include/c++/13/cctype:

/usr/include/c++/13/tr1/gamma.tcc:

/usr/include/x86_64-linux-gnu/bits/posix1_lim.h:

/usr/include/c++/13/debug/assertions.h:

/usr/include/c++/13/cstdlib:

/usr/include/x86_64-linux-gnu/bits/libm-simd-decl-stubs.h:

/usr/include/x86_64-linux-gnu/bits/types/struct_FILE.h:

/usr/include/c++/13/cstddef:

/usr/include/c++/13/bits/stl_algo.h:

/usr/include/c++/13/ostream:

/usr/include/c++/13/tr1/riemann_zeta.tcc:

/usr/include/c++/13/cmath:

/usr/include/c++/13/cerrno:

/home/christianw/raytracer-cuda/include/raytracer/hittable.h:

/usr/include/x86_64-linux-gnu/bits/types/mbstate_t.h:

/usr/include/c++/13/bits/vector.tcc:

/usr/include/c++/13/stdlib.h:

/usr/include/c++/13/bits/stl_function.h:

/usr/include/c++/13/tr1/beta_function.tcc:

/usr/include/c++/13/bits/uses_allocator_args.h:

/usr/include/c++/13/pstl/execution_defs.h:

/usr/include/surface_types.h:

/usr/include/c++/13/exception:

/usr/include/curand_philox4x32_x.h:

/usr/include/c++/13/bits/stl_pair.h:

/usr/include/x86_64-linux-gnu/bits/errno.h:

/usr/include/c++/13/bits/uses_allocator.h:

/usr/include/c++/13/bits/std_abs.h:

/usr/include/x86_64-linux-gnu/bits/mathcalls-narrow.h:

/usr/include/c++/13/bits/stringfwd.h:

/usr/include/endian.h:

/usr/include/c++/13/bits/new_allocator.h:

/usr/include/c++/13/bits/stl_uninitialized.h:

/usr/include/c++/13/bits/stl_vector.h:

/usr/include/c++/13/bits/stl_tempbuf.h:

/usr/include/x86_64-linux-gnu/bits/types/struct_sched_param.h:

/usr/include/asm-generic/errno.h:

/usr/include/x86_64-linux-gnu/bits/types/struct_timeval.h:

/usr/include/c++/13/ext/numeric_traits.h:

/usr/include/c++/13/bits/stl_iterator.h:

/usr/include/c++/13/bits/functexcept.h:

/usr/include/x86_64-linux-gnu/bits/libc-header-start.h:

/usr/include/c++/13/bits/stl_construct.h:

/usr/include/c++/13/bits/stl_bvector.h:

/usr/include/c++/13/bits/allocator.h:

/usr/include/c++/13/bits/stl_algobase.h:

/home/christianw/raytracer-cuda/include/raytracer/material.h:

/usr/include/c++/13/bits/locale_facets.tcc:

/usr/include/c++/13/bits/refwrap.h:

/usr/include/x86_64-linux-gnu/bits/types/__fpos_t.h:

/usr/include/c++/13/stdexcept:

/usr/include/c++/13/system_error:

/usr/include/c++/13/bits/string_view.tcc:

/usr/include/x86_64-linux-gnu/bits/thread-shared-types.h:

/usr/include/c++/13/bits/ptr_traits.h:

/usr/include/c++/13/iosfwd:

/usr/include/c++/13/bits/move.h:

/usr/include/c++/13/algorithm:

/usr/include/curand.h:

/usr/include/c++/13/clocale:

/usr/include/c++/13/bits/postypes.h:

/home/christianw/raytracer-cuda/include/raytracer/render.h:

/home/christianw/raytracer-cuda/include/raytracer/sphere_gpu.h:

/usr/include/c++/13/bits/invoke.h:

/usr/include/x86_64-linux-gnu/sys/select.h:

/usr/include/c++/13/cwchar:

/usr/include/x86_64-linux-gnu/bits/mathcalls-helper-functions.h:

/usr/include/x86_64-linux-gnu/bits/types/error_t.h:

/usr/include/c++/13/bits/hash_bytes.h:

/usr/include/x86_64-linux-gnu/bits/math-vector.h:

/usr/include/x86_64-linux-gnu/bits/types/struct___jmp_buf_tag.h:

/usr/include/x86_64-linux-gnu/bits/types/clockid_t.h:

/home/christianw/raytracer-cuda/include/raytracer/vec3.h:

/usr/include/texture_types.h:

/usr/include/x86_64-linux-gnu/bits/types/locale_t.h:

/usr/include/alloca.h:

/usr/include/c++/13/bits/basic_string.tcc:

/home/christianw/raytracer-cuda/include/raytracer/aabb.h:

/usr/include/c++/13/bit:

/usr/include/c++/13/bits/stl_iterator_base_funcs.h:

/usr/include/asm-generic/errno-base.h:

/home/christianw/raytracer-cuda/include/raytracer/rtweekend.h:

/usr/include/c++/13/bits/exception_ptr.h:

/usr/include/c++/13/bits/istream.tcc:

/home/christianw/raytracer-cuda/include/raytracer/ray.h:

/usr/include/c++/13/bits/stl_relops.h:

/usr/include/x86_64-linux-gnu/bits/mathcalls.h:

/usr/include/c++/13/bits/exception.h:

/usr/include/curand_poisson.h:

/usr/include/x86_64-linux-gnu/bits/stdlib-float.h:

/usr/include/curand_globals.h:

/usr/include/c++/13/ios:

/usr/include/memory.h:

/home/christianw/raytracer-cuda/include/raytracer/cuda_utils.h:

/usr/include/c++/13/bits/cxxabi_init_exception.h:

/usr/include/c++/13/typeinfo:

/usr/include/x86_64-linux-gnu/c++/13/bits/c++allocator.h:

/home/christianw/raytracer-cuda/include/raytracer/random_utils.h:

/usr/include/c++/13/bits/ostream.tcc:

/home/christianw/raytracer-cuda/include/raytracer/camera_data.h:

/usr/include/c++/13/bits/locale_facets.h:

/usr/include/x86_64-linux-gnu/bits/waitstatus.h:

/usr/include/c++/13/bits/exception_defines.h:

/usr/include/nv/detail/__preprocessor:

/usr/include/c++/13/bits/localefwd.h:

/usr/include/cuda_runtime_api.h:

/usr/include/c++/13/bits/streambuf_iterator.h:

/usr/include/x86_64-linux-gnu/bits/flt-eval-method.h:

/usr/include/c++/13/new:

/usr/include/c++/13/ext/alloc_traits.h:

/usr/include/c++/13/bits/memoryfwd.h:

/home/christianw/raytracer-cuda/include/raytracer/color.h:

/usr/include/c++/13/tr1/poly_hermite.tcc:

/usr/include/c++/13/backward/binders.h:

/home/christianw/raytracer-cuda/include/raytracer/hittable_dispatch.h:

/usr/include/features-time64.h:

/home/christianw/raytracer-cuda/include/raytracer/interval.h:

/home/christianw/raytracer-cuda/include/raytracer/instances.h:

/usr/include/x86_64-linux-gnu/bits/atomic_wide_counter.h:

/usr/include/x86_64-linux-gnu/bits/long-double.h:

/usr/include/c++/13/bits/algorithmfwd.h:

/usr/include/x86_64-linux-gnu/bits/fp-logb.h:

/usr/include/c++/13/bits/alloc_traits.h:

/usr/include/c++/13/bits/concept_check.h:

/usr/include/c++/13/bits/basic_ios.h:

/usr/include/x86_64-linux-gnu/bits/time64.h:

/usr/include/crt/host_defines.h:

/usr/include/c++/13/bits/streambuf.tcc:

/usr/include/c++/13/bits/specfun.h:

/usr/include/c++/13/bits/basic_string.h:

/usr/include/c++/13/tr1/poly_laguerre.tcc:

/usr/include/x86_64-linux-gnu/bits/endianness.h:

/usr/include/builtin_types.h:

/usr/include/c++/13/streambuf:

/usr/include/c++/13/bits/memory_resource.h:

/usr/include/wchar.h:

/usr/include/c++/13/bits/cpp_type_traits.h:

/usr/include/x86_64-linux-gnu/bits/types.h:

/usr/include/c++/13/bits/cxxabi_forced.h:

/usr/include/c++/13/bits/functional_hash.h:

/usr/include/errno.h:

/usr/include/c++/13/bits/ios_base.h:

/usr/include/x86_64-linux-gnu/c++/13/bits/ctype_inline.h:

/usr/include/c++/13/bits/char_traits.h:

/usr/lib/gcc/x86_64-linux-gnu/13/include/syslimits.h:

/usr/include/x86_64-linux-gnu/bits/types/clock_t.h:

/usr/include/c++/13/bits/uniform_int_dist.h:

/usr/include/c++/13/bits/locale_classes.tcc:

/usr/include/x86_64-linux-gnu/bits/stdint-intn.h:

/usr/include/c++/13/string:

/usr/include/c++/13/tr1/ell_integral.tcc:

/usr/include/c++/13/tr1/exp_integral.tcc:

/home/christianw/raytracer-cuda/include/raytracer/bvh_builder.h:

/home/christianw/raytracer-cuda/include/raytracer/hittable_dispatch_impl.h:

/usr/include/curand_mtgp32.h:

/usr/include/c++/13/initializer_list:

/usr/include/c++/13/tr1/bessel_function.tcc:

/usr/include/limits.h:

/usr/include/c++/13/bits/range_access.h:

/usr/include/c++/13/tr1/hypergeometric.tcc:

/home/christianw/raytracer-cuda/include/raytracer/quad.h:

/usr/include/c++/13/type_traits:

/usr/include/curand_kernel.h:

/usr/include/c++/13/tr1/legendre_function.tcc:

/usr/lib/gcc/x86_64-linux-gnu/13/include/limits.h:

/usr/include/c++/13/tr1/modified_bessel_func.tcc:

/home/christianw/raytracer-cuda/src/cpu/main.cpp:

/home/christianw/raytracer-cuda/include/raytracer/hittable_list.h:

/usr/include/nv/detail/__target_macros:

/usr/include/c++/13/tr1/special_function_util.h:

/usr/include/stdc-predef.h:

/usr/include/c++/13/bits/nested_exception.h:

/usr/include/c++/13/tuple:

/usr/include/c++/13/bits/stl_heap.h:

/usr/include/c++/13/utility:

/usr/include/c++/13/vector:

/usr/include/crt/host_config.h:

/usr/include/curand_mtgp32_kernel.h:

/usr/include/ctype.h:

/usr/include/cuda_device_runtime_api.h:

/usr/include/x86_64-linux-gnu/bits/types/FILE.h:

/usr/include/cuda_runtime.h:

/usr/include/c++/13/cstdio:

/usr/include/curand_discrete2.h:

/usr/include/curand_lognormal.h:

/usr/include/x86_64-linux-gnu/asm/errno.h:

/usr/include/x86_64-linux-gnu/gnu/stubs-64.h:

/usr/include/curand_mrg32k3a.h:

/usr/include/x86_64-linux-gnu/bits/stdio_lim.h:

/usr/include/x86_64-linux-gnu/bits/timex.h:

/usr/include/curand_normal.h:

/home/christianw/raytracer-cuda/include/raytracer/cuda_compat.h:

/usr/include/vector_functions.h:

/usr/include/curand_uniform.h:

/usr/include/c++/13/cwctype:

/usr/include/driver_functions.h:

/usr/include/curand_precalc.h:

/usr/include/driver_types.h:

/usr/include/features.h:

/usr/include/c++/13/bits/predefined_ops.h:

/usr/include/library_types.h:

/usr/include/c++/13/debug/debug.h:

/usr/include/linux/errno.h:

/usr/include/x86_64-linux-gnu/bits/pthread_stack_min-dynamic.h:

/usr/include/linux/limits.h:

/usr/include/x86_64-linux-gnu/bits/posix2_lim.h:

/usr/include/curand_discrete.h:

/usr/include/locale.h:

/usr/include/c++/13/bits/utility.h:

/usr/include/x86_64-linux-gnu/bits/pthreadtypes-arch.h:

/usr/include/nv/target:

/usr/include/pthread.h:

/usr/include/wctype.h:

/home/christianw/raytracer-cuda/include/raytracer/bvh.h:

/usr/include/device_types.h:

/usr/include/sched.h:

/usr/include/c++/13/bits/stl_iterator_base_types.h:

/usr/include/curand_normal_static.h:

/usr/include/stdio.h:

/home/christianw/raytracer-cuda/include/raytracer/camera.h:

/usr/include/stdlib.h:

/usr/include/string.h:

/usr/include/strings.h:

/usr/include/x86_64-linux-gnu/bits/struct_rwlock.h:

/usr/include/vector_functions.hpp:

/usr/include/vector_types.h:

/usr/include/x86_64-linux-gnu/bits/byteswap.h:

/usr/include/x86_64-linux-gnu/bits/cpu-set.h:

/usr/include/x86_64-linux-gnu/bits/endian.h:

/usr/include/c++/13/bits/requires_hosted.h:

/usr/include/x86_64-linux-gnu/bits/floatn-common.h:

/usr/include/x86_64-linux-gnu/bits/floatn.h:

/usr/include/x86_64-linux-gnu/bits/fp-fast.h:

/usr/include/c++/13/string_view:

/usr/include/x86_64-linux-gnu/bits/iscanonical.h:

/usr/include/x86_64-linux-gnu/bits/local_lim.h:

/usr/include/x86_64-linux-gnu/bits/types/cookie_io_functions_t.h:

/usr/include/x86_64-linux-gnu/bits/locale.h:

/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h:

/usr/include/x86_64-linux-gnu/bits/sched.h:

/usr/include/time.h:

/usr/include/x86_64-linux-gnu/bits/setjmp.h:

/usr/include/x86_64-linux-gnu/bits/types/__fpos64_t.h:

/usr/include/x86_64-linux-gnu/bits/struct_mutex.h:

/usr/include/x86_64-linux-gnu/bits/select.h:

/usr/include/x86_64-linux-gnu/bits/time.h:

/usr/include/c++/13/bits/locale_classes.h:

/usr/include/x86_64-linux-gnu/bits/timesize.h:

/usr/include/math.h:

/usr/include/x86_64-linux-gnu/bits/types/__FILE.h:

/usr/include/x86_64-linux-gnu/bits/types/__locale_t.h:

/usr/include/x86_64-linux-gnu/bits/types/__mbstate_t.h:

/usr/include/c++/13/bits/charconv.h:

/usr/include/x86_64-linux-gnu/bits/types/__sigset_t.h:
