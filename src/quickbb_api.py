import networkx as nx
import subprocess
import logging as log

def gen_cnf(filename, graph):
    v = graph.number_of_nodes()
    e = graph.number_of_edges()
    log.info(f"generating config file {filename}")
    cnf = "c a configuration of -qtree simulator\n"
    cnf += f"p cnf {v} {e}\n"
    for line in nx.generate_edgelist(g):
        cnf+=line.replace("{}",' 0\n')

    #print("cnf file:",cnf)
    with open(filename,'w+') as fp:
        fp.write(cnf)

def run_quickbb(cnffile, command='./quickbb_64'):
    outfile = 'quickbb_out.qbb'
    statfile = 'quickbb_stat.qbb'
    sh = command
    sh += " --min-fill-ordering "
    sh += " --time 60 "
    sh += f" --outfile {outfile} --statfile {statfile} "
    sh += f" --cnffile {cnffile} "
    log.info("excecuting quickbb: "+sh)
    process = subprocess.Popen(
        sh.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    if error:
        log.error(error)
    log.info(output)
    with open(outfile,'r') as fp:
        log.info("OUTPUT:\n"+fp.read())
    with open(statfile,'r') as fp:
        log.info("STAT:\n"+fp.read())

    return output


