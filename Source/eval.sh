#!/bin/sh
# Modify the following variables to match your configuration.

# The path where ground truth binary masks are
GT_PATH=./gt/
# The path where saliency maps computed by your system are
SAL_PATH=./method_DL/
# The location of the saliency evaluation tool
EVAL_TOOL=./SaliencyEvaluationTool/SaliencyEvaluationTool.jar
# The path where to write results to
RES_PATH=./plots/

# This ensures the results directory exists
mkdir -p "${RES_PATH}"

# Run the evaluation script using paths defined above
java -jar ${EVAL_TOOL} pathGT="${GT_PATH}" pathSM="${SAL_PATH}"
	 pathResult="${RES_PATH}"
