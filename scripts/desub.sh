bkill $(bstat | awk 'NR>1{print $1}')
