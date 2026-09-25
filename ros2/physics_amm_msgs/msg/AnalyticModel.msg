# One analytic model (formula) learned by ANSR, as mirrored to disk in a
# current_best_<version>/ snapshot. Structurally parallel to
# coresense_msgs/ModelDescription so a later migration is a field mapping.

string expression              # analytic formula in Python syntax
string simplified_expression   # sympy-simplified form; may be empty
float64 valid_loss             # validation loss; NaN if not yet measured
float64 complexity             # ANSR complexity measure; NaN if unknown
int32 num_active_nodes         # active nodes in the topology; -1 if unknown
float64 rmse_constr            # constraint RMSE; NaN if unknown / no constraints
uint32 dataset_version         # current_best_<version> snapshot this came from
string population              # "archive" (non-dominated set) or "mainpop"
string txt_path                # absolute path to the individual's .txt dump
string m_path                  # absolute path to the MATLAB .m export ("" if absent)
