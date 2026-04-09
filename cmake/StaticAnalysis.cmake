if(SIRIUS_ENABLE_CLANG_TIDY)
  find_program(CLANG_TIDY_EXE NAMES clang-tidy)
  if(CLANG_TIDY_EXE)
    set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE}")
  endif()
endif()

if(SIRIUS_ENABLE_CPPCHECK)
  find_program(CPPCHECK_EXE NAMES cppcheck)
  if(CPPCHECK_EXE)
    set(CMAKE_CXX_CPPCHECK
      "${CPPCHECK_EXE};--enable=warning,performance,portability;--inline-suppr")
  endif()
endif()

if(MSVC)
  add_compile_options($<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/analyze>)
endif()