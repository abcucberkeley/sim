function(target_set_warnings tgt)
  if(MSVC)
    target_compile_options(${tgt} PRIVATE
      $<$<COMPILE_LANGUAGE:C,CXX>:/W4>
      $<$<COMPILE_LANGUAGE:CXX>:/permissive->
    )
  else()
    target_compile_options(${tgt} PRIVATE
      $<$<COMPILE_LANGUAGE:C,CXX>:-Wall>
      $<$<COMPILE_LANGUAGE:C,CXX>:-Wextra>
      $<$<COMPILE_LANGUAGE:C,CXX>:-Wpedantic>
    )
  endif()
endfunction()