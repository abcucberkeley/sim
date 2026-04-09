function(target_enable_sanitizers tgt)
  if(NOT SIRIUS_ENABLE_SANITIZERS)
    return()
  endif()

  if(MSVC)
    target_compile_options(${tgt} PRIVATE
      $<$<CONFIG:Debug>:/fsanitize=address /Zi>
    )
    set_target_properties(${tgt} PROPERTIES MSVC_DEBUG_INFORMATION_FORMAT ProgramDatabase)
    target_link_options(${tgt} PRIVATE
      $<$<CONFIG:Debug>:/INCREMENTAL:NO>
    )
  else()
    target_compile_options(${tgt} PRIVATE
      $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:C,CXX>>:-fsanitize=address,undefined>
    )
    target_link_options(${tgt} PRIVATE
      $<$<CONFIG:Debug>:-fsanitize=address,undefined>
    )
  endif()
endfunction()