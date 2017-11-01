# assyria

You should have a similar directory structure:

     jhwi/
          cleaning_code/
          estimation_directional/
          estimation_nondirectional/
          rawdata/
     parent/
          keys.ini
          estimation/
          process/
                  process.py
                  test_process.py
	      raw/


Where all raw datasets are stored in the raw/ directory.

Sample keys.ini:

     [paths]
     root = /path/to/parent/
     root_jhwi = /path/to/jhwi/
     raw_iticount = raw/anccities_trade_iticount_for_inverse_gravity.csv
     raw_city_name = raw/city_name.csv
     raw_coord = raw/coordinates.csv
     raw_constr_dynamic = raw/ancient_constraints_dynamic.csv
     raw_constr_static = raw/ancient_constraints_static.csv
     raw_constr_format = raw/constraint_format.csv
     process = process/
