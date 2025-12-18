from web3 import Web3
from eth_account import Account
import json
import os
from typing import Dict, Any
from .ethereum import EthereumNetwork

class PolygonNetwork(EthereumNetwork):
    """Polygon (Matic) blockchain integration"""
    
    def __init__(self, network='mainnet', config_path='../../config.yaml'):
        import yaml
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.network = network
        self.network_config = self.config['blockchain_networks']['polygon'][network]
        
        # Initialize Web3 with Polygon RPC
        self.w3 = Web3(Web3.HTTPProvider(self.network_config['rpc_url']))
        
        # Load account (can use same as Ethereum with proper derivation)
        self.private_key = os.getenv(f'POLYGON_{network.upper()}_PRIVATE_KEY') or os.getenv('ETHEREUM_PRIVATE_KEY')
        if self.private_key:
            self.account = Account.from_key(self.private_key)
            self.address = self.account.address
        else:
            self.account = None
            self.address = None
        
        # Polygon-specific settings
        self.gas_limit = 3000000  # Higher gas limit for Polygon
        self.max_priority_fee = Web3.to_wei('30', 'gwei')  # Polygon has different fee structure
        self.max_fee = Web3.to_wei('100', 'gwei')
        
        # Contract
        self.contract_address = self.network_config.get('contract_address')
        self.contract_abi = self._load_contract_abi()
    
    def get_gas_price(self):
        """Get gas price for Polygon (different method)"""
        try:
            # Try Polygon gas station API
            response = requests.get('https://gasstation.polygon.technology/v2')
            data = response.json()
            
            # Use fast gas price
            gas_price = Web3.to_wei(data['fast']['maxPriorityFee'], 'gwei')
            return gas_price
            
        except:
            # Fallback to standard method
            return super().get_gas_price()
    
    def bridge_to_ethereum(self, amount, ethereum_address):
        """Bridge assets from Polygon to Ethereum"""
        # This would interact with Polygon bridge contracts
        # Implementation depends on specific bridge (PoS bridge, Plasma, etc.)
        pass
    
    def register_model_with_fee(self, model_hash, metadata_hash, version, ipfs_cid='', fee_amount=None):
        """Register model with MATIC fee payment"""
        if fee_amount is None:
            # Calculate fee based on current gas prices
            fee_amount = Web3.to_wei('0.01', 'ether')  # 0.01 MATIC default
        
        # First transfer fee to contract if needed
        # Then call registerModel
        pass