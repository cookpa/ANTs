if(NOT DEFINED CREATE_IMAGE OR NOT DEFINED OPTIMIZER OR NOT DEFINED TEST_OUTPUT_DIR)
  message(FATAL_ERROR "The test requires CREATE_IMAGE, OPTIMIZER, and TEST_OUTPUT_DIR.")
endif()

file(MAKE_DIRECTORY "${TEST_OUTPUT_DIR}")
set(input_image "${TEST_OUTPUT_DIR}/antsOptimizeImageSetAppearanceInput.nii.gz")
set(second_input_image "${TEST_OUTPUT_DIR}/antsOptimizeImageSetAppearanceSecondInput.nii.gz")
set(mismatched_image "${TEST_OUTPUT_DIR}/antsOptimizeImageSetAppearanceMismatchedInput.nii.gz")
set(output_image "${TEST_OUTPUT_DIR}/antsOptimizeImageSetAppearanceIdentity.nii.gz")
set(otsu_output_image "${TEST_OUTPUT_DIR}/antsOptimizeImageSetAppearanceOtsu.nii.gz")

execute_process(
  COMMAND "${CREATE_IMAGE}" 2 "${input_image}" 0x0 1x1 32x32 255 1
  RESULT_VARIABLE create_result
  OUTPUT_VARIABLE create_output
  ERROR_VARIABLE create_error
  )
if(NOT create_result EQUAL 0)
  message(FATAL_ERROR "CreateImage failed:\n${create_output}\n${create_error}")
endif()

execute_process(
  COMMAND "${CREATE_IMAGE}" 2 "${second_input_image}" 0x0 1x1 32x32 255 1
  RESULT_VARIABLE create_second_result
  OUTPUT_VARIABLE create_second_output
  ERROR_VARIABLE create_second_error
  )
if(NOT create_second_result EQUAL 0)
  message(FATAL_ERROR
    "CreateImage failed for the second input:\n${create_second_output}\n${create_second_error}")
endif()

execute_process(
  COMMAND "${CREATE_IMAGE}" 2 "${mismatched_image}" 0x0 1x1 24x32 255 1
  RESULT_VARIABLE create_mismatched_result
  OUTPUT_VARIABLE create_mismatched_output
  ERROR_VARIABLE create_mismatched_error
  )
if(NOT create_mismatched_result EQUAL 0)
  message(FATAL_ERROR
    "CreateImage failed for the mismatched input:\n${create_mismatched_output}\n${create_mismatched_error}")
endif()

execute_process(
  COMMAND "${OPTIMIZER}"
    -d 2
    -i "${input_image}"
    -i "${mismatched_image}"
    -o "${output_image}"
  RESULT_VARIABLE mismatched_result
  OUTPUT_QUIET
  ERROR_QUIET
  )
if(mismatched_result EQUAL 0)
  message(FATAL_ERROR "antsOptimizeImageSetAppearance accepted inputs with mismatched geometry.")
endif()

execute_process(
  COMMAND "${OPTIMIZER}"
    -d 2
    -i "${input_image}"
    -i "${input_image}"
    -o "${output_image}"
    -c 2
    -r 2
    -g 0.1
    -s 0
  RESULT_VARIABLE optimize_result
  OUTPUT_VARIABLE optimize_output
  ERROR_VARIABLE optimize_error
  )
if(NOT optimize_result EQUAL 0)
  message(FATAL_ERROR
    "antsOptimizeImageSetAppearance failed:\n${optimize_output}\n${optimize_error}")
endif()
if(NOT EXISTS "${output_image}")
  message(FATAL_ERROR "antsOptimizeImageSetAppearance did not create ${output_image}.")
endif()

execute_process(
  COMMAND "${OPTIMIZER}"
    -d 2
    -i "${input_image}"
    -i "${second_input_image}"
    -o "${otsu_output_image}"
    -c 5
    -r 2
    -g 0.25
    -s 0
    -x Otsu
    -v 1
  RESULT_VARIABLE otsu_result
  OUTPUT_VARIABLE otsu_output
  ERROR_VARIABLE otsu_error
  )
if(NOT otsu_result EQUAL 0)
  message(FATAL_ERROR
    "Otsu-masked antsOptimizeImageSetAppearance failed:\n${otsu_output}\n${otsu_error}")
endif()
if(NOT EXISTS "${otsu_output_image}")
  message(FATAL_ERROR "antsOptimizeImageSetAppearance did not create ${otsu_output_image}.")
endif()
if(NOT otsu_output MATCHES "Iteration 1:.*step =")
  message(FATAL_ERROR "The Otsu-masked appearance test did not accept an improving update:\n${otsu_output}")
endif()
