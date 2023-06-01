#!/usr/bin/env python3

import pytraj as pt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

# Create an argument parser
def get_args():
	parser = argparse.ArgumentParser(description='Process simulation files')
	parser.add_argument('-p', '--prmtop', action='store',  help='AMBER parameter file (.prmtop)', nargs='+')
	parser.add_argument('-nc', '--binpos', action='store', help='AMBER trajectory file (.binpos)', nargs='+')
	
	return parser.parse_args()

args =  get_args() 

# Load trajectory using the provided files
traj2 = pt.load(args.binpos, top=args.prmtop)

dist = pt.distance(traj2, [':208@OD1 :345@N', ':253@OD1 :345@NH1', ':253@OD2 :345@NH2', ':319@NZ :345@C'], dtype='dataframe')

dist.rename(columns={'Dis_00000':'Asp200', 'Dis_00001':'Asp245_O1', 'Dis_00002':'Asp245_O2', 'Dis_00003':'Lys311'}, inplace=True)

dist_melt = pd.melt(dist)
#print(dist_melt)

#make a graph 

dist_melt.rename(columns={'variable':'Residues', 'value':'Bond Distance'}, inplace=True)
sns.violinplot(data=dist_melt, x='Residues', y='Bond Distance', palette=['b', 'r', 'm', 'c'])

plt.xlabel('Residue Index')
plt.ylabel(r"Bond Distance ($\AA$)")

plt.savefig('Sub_BondDistance.png', bbox_inches='tight', dpi=300)
