from mstid import mongo_tools
import datetime
# am, qm = mongo_tools.get_mstid_days()
ml =mongo_tools.get_active_list()
dl = mongo_tools.loadDayLists(ml)
print(ml)
print(dl)