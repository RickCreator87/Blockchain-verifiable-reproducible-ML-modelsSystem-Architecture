from web3 import Web3
from eth_account import Account
from eth_account.messages import encode_defunct
import json
import os
from typing import Dict, Any, Optional
from pathlib import Path
import requests

class EthereumNetwork:
    """Ethereum blockchain integration with support for multiple networks"""
    
    def __init__(self, network='mainnet', config_path='../../config.yaml'):
        import yaml
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.network = network
        self.network_config = self.config['blockchain_networks']['ethereum'][network]
        
        # Initialize Web3
        self.w3 = Web3(Web3.HTTPProvider(self.network_config['rpc_url']))
        
        # Load account
        self.private_key = os.getenv(f'ETHEREUM_{network.upper()}_PRIVATE_KEY')
        if self.private_key:
            self.account = Account.from_key(self.private_key)
            self.address = self.account.address
        else:
            self.account = None
            self.address = None
        
        # Contract ABI and address
        self.contract_address = self.network_config.get('contract_address')
        self.contract_abi = self._load_contract_abi()
        
        # Gas settings
        self.gas_limit = self.config['blockchain']['gas_limit']
        self.max_priority_fee = Web3.to_wei('2', 'gwei')
        self.max_fee = Web3.to_wei('100', 'gwei')
    
    def _load_contract_abi(self):
        """Load contract ABI from file"""
        abi_path = Path(__file__).parent.parent / 'contracts' / 'abis' / 'ModelRegistry.json'
        if abi_path.exists():
            with open(abi_path, 'r') as f:
                return json.load(f)
        return None
    
    def is_connected(self):
        """Check if connected to Ethereum network"""
        return self.w3.is_connected()
    
    def get_balance(self, address=None):
        """Get ETH balance"""
        if address is None:
            address = self.address
        balance = self.w3.eth.get_balance(address)
        return self.w3.from_wei(balance, 'ether')
    
    def get_gas_price(self):
        """Get current gas price"""
        return self.w3.eth.gas_price
    
    def get_nonce(self, address=None):
        """Get transaction nonce"""
        if address is None:
            address = self.address
        return self.w3.eth.get_transaction_count(address)
    
    def sign_message(self, message):
        """Sign a message with private key"""
        if not self.account:
            raise ValueError("No account configured")
        
        message_hash = encode_defunct(text=message)
        signed_message = self.w3.eth.account.sign_message(message_hash, private_key=self.private_key)
        return signed_message.signature.hex()
    
    def verify_signature(self, message, signature, address):
        """Verify a signed message"""
        message_hash = encode_defunct(text=message)
        recovered_address = self.w3.eth.account.recover_message(message_hash, signature=signature)
        return recovered_address.lower() == address.lower()
    
    def deploy_contract(self, contract_source_path, constructor_args=()):
        """Deploy smart contract to Ethereum"""
        from solcx import compile_source
        
        # Compile contract
        with open(contract_source_path, 'r') as f:
            contract_source = f.read()
        
        compiled = compile_source(
            contract_source,
            output_values=['abi', 'bin'],
            solc_version='0.8.19'
        )
        
        contract_id, contract_interface = compiled.popitem()
        
        # Get contract
        contract = self.w3.eth.contract(
            abi=contract_interface['abi'],
            bytecode=contract_interface['bin']
        )
        
        # Build transaction
        transaction = contract.constructor(*constructor_args).build_transaction({
            'from': self.address,
            'nonce': self.get_nonce(),
            'gas': self.gas_limit,
            'gasPrice': self.get_gas_price(),
            'chainId': self.network_config['chain_id']
        })
        
        # Sign and send
        signed_txn = self.w3.eth.account.sign_transaction(transaction, self.private_key)
        tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
        
        # Wait for receipt
        tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        
        contract_address = tx_receipt.contractAddress
        
        # Save deployment info
        deployment_info = {
            'contract_address': contract_address,
            'transaction_hash': tx_hash.hex(),
            'block_number': tx_receipt.blockNumber,
            'network': self.network,
            'deployer': self.address,
            'timestamp': self.w3.eth.get_block(tx_receipt.blockNumber).timestamp
        }
        
        # Update config
        self.network_config['contract_address'] = contract_address
        self._update_config()
        
        return contract_address, contract_interface['abi']
    
    def register_model(self, model_hash, metadata_hash, version, ipfs_cid=''):
        """Register model on Ethereum blockchain"""
        if not self.contract_address or not self.contract_abi:
            raise ValueError("Contract not deployed")
        
        # Get contract instance
        contract = self.w3.eth.contract(
            address=self.contract_address,
            abi=self.contract_abi
        )
        
        # Build transaction
        transaction = contract.functions.registerModel(
            model_hash,
            metadata_hash,
            version,
            ipfs_cid
        ).build_transaction({
            'from': self.address,
            'nonce': self.get_nonce(),
            'gas': self.gas_limit,
            'gasPrice': self.get_gas_price(),
            'chainId': self.network_config['chain_id']
        })
        
        # Sign and send
        signed_txn = self.w3.eth.account.sign_transaction(transaction, self.private_key)
        tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
        
        # Wait for confirmation
        self.w3.eth.wait_for_transaction_receipt(tx_hash)
        
        return tx_hash.hex()
    
    def verify_model(self, model_hash, verifier_address=None):
        """Verify a model on blockchain"""
        contract = self.w3.eth.contract(
            address=self.contract_address,
            abi=self.contract_abi
        )
        
        if verifier_address is None:
            verifier_address = self.address
        
        transaction = contract.functions.verifyModel(model_hash).build_transaction({
            'from': verifier_address,
            'nonce': self.get_nonce(verifier_address),
            'gas': self.gas_limit,
            'gasPrice': self.get_gas_price(),
            'chainId': self.network_config['chain_id']
        })
        
        signed_txn = self.w3.eth.account.sign_transaction(transaction, self.private_key)
        tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
        
        return tx_hash.hex()
    
    def get_model_info(self, model_hash):
        """Get model information from blockchain"""
        contract = self.w3.eth.contract(
            address=self.contract_address,
            abi=self.contract_abi
        )
        
        try:
            record = contract.functions.getModelRecord(model_hash).call()
            
            return {
                'model_hash': record[0],
                'metadata_hash': record[1],
                'owner': record[2],
                'timestamp': record[3],
                'block_number': record[4],
                'version': record[5],
                'ipfs_cid': record[6],
                'verified': record[7]
            }
        except:
            return None
    
    def get_models_by_owner(self, owner_address):
        """Get all models owned by an address"""
        contract = self.w3.eth.contract(
            address=self.contract_address,
            abi=self.contract_abi
        )
        
        try:
            model_hashes = contract.functions.getModelsByOwner(owner_address).call()
            return model_hashes
        except:
            return []
    
    def listen_for_events(self, event_name, from_block=0, to_block='latest'):
        """Listen for smart contract events"""
        contract = self.w3.eth.contract(
            address=self.contract_address,
            abi=self.contract_abi
        )
        
        event_filter = contract.events[event_name].create_filter(
            fromBlock=from_block,
            toBlock=to_block
        )
        
        return event_filter.get_all_entries()
    
    def _update_config(self):
        """Update configuration file with new contract address"""
        import yaml
        
        config_path = Path(__file__).parent.parent.parent / 'config.yaml'
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        config['blockchain_networks']['ethereum'][self.network]['contract_address'] = \
            self.network_config['contract_address']
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

# ERC-20 and ERC-721 Token Support
class EthereumTokens:
    """Ethereum token utilities"""
    
    @staticmethod
    def create_erc20_token(w3, private_key, token_name, token_symbol, total_supply):
        """Create ERC-20 token for model marketplace"""
        erc20_abi = [
            {
                "constant": False,
                "inputs": [
                    {"name": "_name", "type": "string"},
                    {"name": "_symbol", "type": "string"},
                    {"name": "_initialSupply", "type": "uint256"}
                ],
                "name": "constructor",
                "outputs": [],
                "payable": False,
                "stateMutability": "nonpayable",
                "type": "constructor"
            }
        ]
        
        # Simple ERC-20 implementation
        erc20_bytecode = "0x608060405234801561001057600080fd5b506040516..."
        
        account = Account.from_key(private_key)
        
        contract = w3.eth.contract(abi=erc20_abi, bytecode=erc20_bytecode)
        
        transaction = contract.constructor(
            token_name, token_symbol, total_supply
        ).build_transaction({
            'from': account.address,
            'nonce': w3.eth.get_transaction_count(account.address),
            'gas': 2000000,
            'gasPrice': w3.eth.gas_price
        })
        
        signed_txn = w3.eth.account.sign_transaction(transaction, private_key)
        tx_hash = w3.eth.send_raw_transaction(signed_txn.rawTransaction)
        
        return tx_hash.hex()
    
    @staticmethod
    def create_model_nft(w3, private_key, model_hash, metadata_uri):
        """Create NFT for model ownership"""
        # ERC-721 NFT for model representation
        nft_abi = [
            {
                "constant": False,
                "inputs": [
                    {"name": "to", "type": "address"},
                    {"name": "tokenId", "type": "uint256"},
                    {"name": "tokenURI", "type": "string"}
                ],
                "name": "mint",
                "outputs": [],
                "payable": False,
                "stateMutability": "nonpayable",
                "type": "function"
            }
        ]
        
        # Implementation would include full ERC-721 contract
        pass