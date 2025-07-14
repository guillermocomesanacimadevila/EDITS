#!/bin/bash
echo "WARNING: This will delete all pipeline outputs, models, logs, and work folders!"
read -p "Continue? (y/n): " confirm
if [[ $confirm == "y" ]]; then
    rm -rf runs/ work/ Data/toy_data_cellflow_* *.log
    echo "Cleanup done."
else
    echo "Aborted."
fi

