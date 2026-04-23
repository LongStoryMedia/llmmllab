

import { InputItem } from './InputItem';

import { Metadata } from './Metadata';



export interface CreateConversationBody {
  items?: (InputItem)[] | unknown;
  metadata?: Metadata | unknown;
}