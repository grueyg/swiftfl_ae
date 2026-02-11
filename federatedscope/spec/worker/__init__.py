from federatedscope.register import register_worker
from federatedscope.core.workers import Client
from federatedscope.spec.worker.spec_client import SpecClient
from federatedscope.spec.worker.spec_server import SpecServer
from federatedscope.spec.worker.base_worker import baseServer, baseClient
from federatedscope.spec.worker.idpspec_server import idpSpecServer
from federatedscope.spec.worker.async_worker import AsyncClient
from federatedscope.spec.exp.oort.oort_worker import OortClient, OortServer
from federatedscope.spec.exp.fednova.fednova_client import FedNovaClient
from federatedscope.spec.exp.scaffold.scaffold_client import ScaffoldClient
from federatedscope.spec.exp.scaffold.scaffold_server import ScaffoldServer
from federatedscope.spec.exp.fjord.fjord_client import FjORDClient
from federatedscope.spec.exp.fjord.fjord_server import FjORDServer
from federatedscope.spec.exp.fluid.fluid_client import FLuIDClient
from federatedscope.spec.exp.fluid.fluid_server import FLuIDServer

def call_spec_worker(method):
    if method in ['fedavg', 'papaya', 'fedadam', 'fedyogi'] or 'debug' in method:
        worker_builder = {'client': baseClient, 'server': baseServer}
    elif method == 'fedasync':
        worker_builder = {'client': AsyncClient, 'server': baseServer}
    elif method == 'pyramidfl':
        worker_builder = {'client': OortClient, 'server': OortServer}
    elif method == 'spec':
        worker_builder = {'client': SpecClient, 'server': SpecServer}
    elif method == 'idpspec':
        worker_builder = {'client': SpecClient, 'server': idpSpecServer}
    elif method == 'fednova':
        worker_builder = {'client': FedNovaClient, 'server': baseServer}
    elif method == 'scaffold':
        worker_builder = {'client': ScaffoldClient, 'server': ScaffoldServer}
    elif method == 'fjord':
        worker_builder = {'client': FjORDClient, 'server': FjORDServer}
    elif method == 'fluid':
        worker_builder = {'client': FLuIDClient, 'server': FLuIDServer}
    else:
        worker_builder = None
    return worker_builder

register_worker('base_spec', call_spec_worker)

