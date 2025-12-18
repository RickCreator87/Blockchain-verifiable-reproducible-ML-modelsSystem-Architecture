import base64
import json
from typing import Dict, Any, Optional
import os
from solders.pubkey import Pubkey
from solders.keypair import Keypair
from solana.rpc.api import Client
from solana.transaction import Transaction
from solana.system_program import transfer, TransferParams
import requests

class SolanaNetwork:
    """Solana blockchain integration"""
    
    def __init__(self, network='mainnet', config_path='../../config.yaml'):
        import yaml
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.network = network
        self.network_config = self.config['blockchain_networks']['solana'][network]
        
        # Initialize Solana client
        self.client = Client(self.network_config['rpc_url'])
        
        # Load keypair
        private_key = os.getenv(f'SOLANA_{network.upper()}_PRIVATE_KEY')
        if private_key:
            # Convert from base58 or bytes
            if private_key.startswith('['):
                # Array format
                key_bytes = json.loads(private_key)
                self.keypair = Keypair.from_bytes(bytes(key_bytes))
            else:
                # Base58 format
                self.keypair = Keypair.from_base58_string(private_key)
            self.public_key = self.keypair.pubkey()
        else:
            self.keypair = None
            self.public_key = None
    
    def is_connected(self):
        """Check connection to Solana"""
        try:
            version = self.client.get_version()
            return version is not None
        except:
            return False
    
    def get_balance(self, public_key=None):
        """Get SOL balance"""
        if public_key is None:
            public_key = self.public_key
        
        balance = self.client.get_balance(public_key)
        return balance.value / 1e9  # Convert lamports to SOL
    
    def create_account(self):
        """Create new Solana account"""
        new_keypair = Keypair()
        
        # Convert to exportable format
        private_key_bytes = bytes(new_keypair)
        private_key_base58 = str(new_keypair)
        
        return {
            'public_key': str(new_keypair.pubkey()),
            'private_key_base58': private_key_base58,
            'private_key_bytes': list(private_key_bytes)
        }
    
    def transfer_sol(self, to_public_key, amount):
        """Transfer SOL to another account"""
        if not self.keypair:
            raise ValueError("No keypair configured")
        
        # Convert SOL to lamports
        lamports = int(amount * 1e9)
        
        # Create transfer instruction
        transfer_instruction = transfer(
            TransferParams(
                from_pubkey=self.public_key,
                to_pubkey=Pubkey.from_string(to_public_key),
                lamports=lamports
            )
        )
        
        # Create and send transaction
        transaction = Transaction().add(transfer_instruction)
        
        # Sign and send
        result = self.client.send_transaction(transaction, self.keypair)
        return result.value
    
    def deploy_program(self, program_path):
        """Deploy Solana program (smart contract)"""
        # Load program from file
        with open(program_path, 'rb') as f:
            program_data = f.read()
        
        # Create buffer for program data
        from solana.rpc.commitment import Confirmed
        from solana.rpc.types import TxOpts
        
        # Deploy program
        result = self.client.deploy_program(
            program_data,
            self.keypair,
            opts=TxOpts(
                skip_confirmation=False,
                preflight_commitment=Confirmed
            )
        )
        
        return result.value
    
    def register_model(self, model_hash, metadata_hash, version):
        """Register model on Solana using custom program"""
        # This would call a deployed Solana program
        # Implementation depends on program ABI/IDL
        
        # For now, store in account data
        model_data = {
            'model_hash': model_hash,
            'metadata_hash': metadata_hash,
            'version': version,
            'owner': str(self.public_key)
        }
        
        # Convert to bytes
        data_bytes = json.dumps(model_data).encode()
        
        # Create account to store data
        # In production, you'd use a proper program
        return self._store_in_account(data_bytes)
    
    def _store_in_account(self, data):
        """Store data in Solana account"""
        # Create new account with enough space
        from solana.system_program import create_account, CreateAccountParams
        from solana.sysvar import SYSVAR_RENT_PUBKEY
        
        # Calculate space needed
        space = len(data)
        lamports = self.client.get_minimum_balance_for_rent_exemption(space).value
        
        # Create account instruction
        new_account = Keypair()
        
        create_account_instruction = create_account(
            CreateAccountParams(
                from_pubkey=self.public_key,
                new_account_pubkey=new_account.pubkey(),
                lamports=lamports,
                space=space,
                program_id=Pubkey.from_string("11111111111111111111111111111111")  # System program
            )
        )
        
        # Create transaction
        transaction = Transaction().add(create_account_instruction)
        
        # Send transaction
        result = self.client.send_transaction(transaction, self.keypair, new_account)
        
        # Store data in account
        # This requires additional instructions
        
        return str(new_account.pubkey())
    
    def get_model_info(self, account_address):
        """Get model information from Solana account"""
        try:
            account_info = self.client.get_account_info(Pubkey.from_string(account_address))
            
            if account_info.value:
                data = account_info.value.data
                model_data = json.loads(data.decode())
                return model_data
        except:
            pass
        
        return None