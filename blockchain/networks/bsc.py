from web3 import Web3
from eth_account import Account
import json
import os
from .ethereum import EthereumNetwork

class BSCNetwork(EthereumNetwork):
    """Binance Smart Chain integration"""
    
    def __init__(self, network='mainnet', config_path='../../config.yaml'):
        import yaml
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.network = network
        self.network_config = self.config['blockchain_networks']['binance_smart_chain'][network]
        
        # Initialize Web3 with BSC RPC
        self.w3 = Web3(Web3.HTTPProvider(self.network_config['rpc_url']))
        
        # BSC uses same address format as Ethereum
        self.private_key = os.getenv(f'BSC_{network.upper()}_PRIVATE_KEY') or os.getenv('ETHEREUM_PRIVATE_KEY')
        if self.private_key:
            self.account = Account.from_key(self.private_key)
            self.address = self.account.address
        else:
            self.account = None
            self.address = None
        
        # BSC-specific settings
        self.gas_limit = 5000000  # BSC has lower gas limits
        self.max_priority_fee = Web3.to_wei('5', 'gwei')  # BSC gas is cheaper
        self.max_fee = Web3.to_wei('20', 'gwei')
        
        # Contract
        self.contract_address = self.network_config.get('contract_address')
        self.contract_abi = self._load_contract_abi('BSC')
    
    def _load_contract_abi(self, network_type='BSC'):
        """Load BSC-specific contract ABI"""
        abi_path = Path(__file__).parent.parent / 'contracts' / 'abis' / f'ModelRegistry{network_type}.json'
        if abi_path.exists():
            with open(abi_path, 'r') as f:
                return json.load(f)
        return None
    
    def get_gas_price(self):
        """Get BSC gas price from BSC Gas Station"""
        try:
            response = requests.get('https://api.bscscan.com/api?module=gastracker&action=gasoracle')
            data = response.json()
            
            if data['status'] == '1':
                gas_price = Web3.to_wei(data['result']['FastGasPrice'], 'gwei')
                return gas_price
        except:
            pass
        
        return super().get_gas_price()
    
    def swap_bnb_for_token(self, token_address, bnb_amount, slippage=0.5):
        """Swap BNB for tokens using PancakeSwap"""
        # This would interact with PancakeSwap router
        # Implementation requires PancakeSwap router ABI
        pass