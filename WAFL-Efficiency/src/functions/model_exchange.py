import copy
import os
import torch
import torch.nn as nn
import sys
import time

def topk_ds(net, contact, fl_coefficient, K, prev_net, zero_initialization, init_net, private_param, device):

	node_num = len(net)
	local_model = [{} for _ in range(node_num)]

	for n in range(node_num): # recv

		local_model[n] = copy.deepcopy(net[n].state_dict())
		nbr = contact[str(n)]  # the nodes n-th node contacted
		n_nbr = len(nbr)

		if n_nbr == 0:
			continue

		# model aggregate (pre process)
		for key in local_model[n]:
			if key in private_param:
				continue
			local_model[n][key] -= fl_coefficient * n_nbr / (n_nbr + 1) * local_model[n][key]

		# save, load, setup prev_net
		for k in range(node_num):
			if k not in nbr:
				if prev_net[k][n] != "saved" and prev_net[k][n] != None: # save
					torch.save(prev_net[k][n], f"../tmp/prev_net_{k}_{n}.pth") 
					prev_net[k][n] = "saved"
			else:
				if prev_net[k][n] == "saved": # load
					prev_net[k][n] = torch.load(f"../tmp//prev_net_{k}_{n}.pth")
				elif prev_net[k][n] == None: # setup (first contact)
					if zero_initialization == True:
						prev_net[k][n] = copy.deepcopy(init_net)
					else:
						prev_net[k][n] = copy.deepcopy(net[k]).to(torch.device("cpu")).state_dict()

		for k in nbr: # send
			param = copy.deepcopy(net[k].state_dict())

			# prev_net to cuda
			for key in prev_net[k][n]:
				if key in private_param:
					continue
				prev_net[k][n][key] = prev_net[k][n][key].to(device)

			# topk ds
			diff = {key: value for key, value in param.items() if key not in private_param}
			for key in diff.keys():
				if key in private_param:
					continue
				diff[key] = param[key] - prev_net[k][n][key]
			diff_threshold = topk_threshold(diff, K, device)
			for key in diff.keys():
				mask = (-1 * diff_threshold < diff[key]) & (diff[key] < diff_threshold)
				diff[key][mask] = 0
				param[key] = prev_net[k][n][key] + diff[key]
			prev_net[k][n] = copy.deepcopy(param)

			# prev_net to cpu
			for key in prev_net[k][n]:
				if key in private_param:
					continue
				prev_net[k][n][key] = prev_net[k][n][key].to(torch.device("cpu"))

			# model aggregate (net[k]'s param)
			for key in local_model[n]:
				if key in private_param:
					continue
				local_model[n][key] += fl_coefficient / (n_nbr + 1) * param[key]

	# update nets
	for n in range(node_num):
		nbr = contact[str(n)]
		if len(nbr) > 0:
			net[n].load_state_dict(local_model[n])


def topk_ds_dq(net, contact, fl_coefficient, K, Q, prev_net, zero_initialization, init_net, private_param, device):

	node_num = len(net)
	local_model = [{} for _ in range(node_num)]

	for n in range(node_num): # recv

		local_model[n] = copy.deepcopy(net[n].state_dict())
		nbr = contact[str(n)]  # the nodes n-th node contacted
		n_nbr = len(nbr)

		if n_nbr == 0:
			continue

		# model aggregate (pre process)
		for key in local_model[n]:
			if key in private_param:
				continue
			local_model[n][key] -= fl_coefficient * n_nbr / (n_nbr + 1) * local_model[n][key]

		# save, load, setup prev_net
		for k in range(node_num):
			if k not in nbr:
				if prev_net[k][n] != "saved" and prev_net[k][n] != None: # save
					torch.save(prev_net[k][n], f"../tmp/prev_net_{k}_{n}.pth") 
					prev_net[k][n] = "saved"
			else:
				if prev_net[k][n] == "saved": # load
					prev_net[k][n] = torch.load(f"../tmp//prev_net_{k}_{n}.pth")
				elif prev_net[k][n] == None: # setup (first contact)
					if zero_initialization == True:
						prev_net[k][n] = copy.deepcopy(init_net)
					else:
						prev_net[k][n] = copy.deepcopy(net[k]).to(torch.device("cpu")).state_dict()

		for k in nbr: # send
			param = copy.deepcopy(net[k].state_dict())

			# prev_net to cuda
			for key in prev_net[k][n]:
				if key in private_param:
					continue
				prev_net[k][n][key] = prev_net[k][n][key].to(device)

			# topk ds
			diff = {key: value for key, value in param.items() if key not in private_param}
			for key in diff.keys():
				if key in private_param:
					continue
				diff[key] = param[key] - prev_net[k][n][key]
			diff_threshold = topk_threshold(diff, K, device)

			for key, value in diff.items():
				if key in private_param:
					continue
				mask = (-1 * diff_threshold < value) & (value < diff_threshold)
				value[mask] = 0
				max_diff = torch.max(value.abs())
				if torch.all(value == 0):
					min_diff = max_diff
				else:
					min_diff = torch.min(value[value != 0].abs())
				if max_diff != min_diff:
					if Q >= 2:
						step = (max_diff - min_diff) / (2 ** (Q - 1) - 1)  # use 1bit for sign
						diff_sign = torch.sign(value)
						diff_qval = torch.round((value.abs() - min_diff) / step) * step + min_diff
						diff[key] = diff_sign * diff_qval
					elif Q == 1:
						diff_sign = torch.sign(value)
						if torch.all(value == 0):
							diff_qval = 0
						else:
							diff_qval = torch.mean(torch.abs(value[value != 0]))
						diff[key] = diff_sign * diff_qval

				param[key] = prev_net[k][n][key] + diff[key]
			prev_net[k][n] = copy.deepcopy(param)

			# prev_net to cpu
			for key in prev_net[k][n]:
				if key in private_param:
					continue
				prev_net[k][n][key] = prev_net[k][n][key].to(torch.device("cpu"))

			# model aggregate (net[k]'s param)
			for key in local_model[n]:
				if key in private_param:
					continue
				local_model[n][key] += fl_coefficient / (n_nbr + 1) * param[key]

	# update nets
	for n in range(node_num):
		nbr = contact[str(n)]
		if len(nbr) > 0:
			net[n].load_state_dict(local_model[n])


def default(net, contact, fl_coefficient, private_param, device):
	node_num = len(net)
	local_model = [{} for _ in range(node_num)]
	for n in range(node_num):

		nbr = contact[str(n)]  # the nodes n-th node contacted
		n_nbr = len(nbr)  # how many nodes n-th node contacted
		if n_nbr == 0:
			continue

		local_model[n] = net[n].state_dict()
		recv_models = []
		for k in nbr:
			param = copy.deepcopy(net[k].state_dict())
			recv_models.append(param)  

		for k in range(n_nbr):
			for key in recv_models[k]:
				if key in private_param:
					continue
				recv_models[k][key] = recv_models[k][key] - local_model[n][key]

		for k in range(n_nbr):
			for key in recv_models[k]:
				if key in private_param:
					continue
				local_model[n][key] += (
					recv_models[k][key] * fl_coefficient / float(n_nbr + 1)
				)

	# update nets
	for n in range(node_num):
		nbr = contact[str(n)]
		if len(nbr) > 0:
			net[n].load_state_dict(local_model[n])

def topk_threshold(net, K, device):

	# the number of parameters 
	full_size = 0
	for key in net:
		full_size += torch.numel(net[key])
	select_size = int(full_size * K)

	params = torch.empty(0).to(device)
	for key in net:
		params = torch.concat((params, torch.abs(torch.flatten(net[key]))))
	params, indices = torch.topk(params, select_size)
	
	return torch.min(params)


def update_nets(net, 
				contact,
				fl_coefficient,
				reduce=None,
				K=1,
				Q=8,
				prev_net=None,
				pruned_net=None,
				epoch=None,
				sparse_aggregation=False,
				K_list=None,
				sending_net=None,
				diff_mean_prev=None,
				threshold=None, 
				K_limit=None,
				zero_initialization = False,
				init_net = None,
				private_param = [],
				device=torch.device("cpu")):
	if reduce == "topk_ds":
		topk_ds(net, contact, fl_coefficient, K, prev_net, zero_initialization, init_net, private_param, device)
	elif reduce == "topk_ds_dq":
		topk_ds_dq(net, contact, fl_coefficient, K, Q, prev_net, zero_initialization, init_net, private_param, device)
	else:
		default(net, contact, fl_coefficient, private_param, device)


